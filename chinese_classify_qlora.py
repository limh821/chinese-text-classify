'''
    以下是一个完整的代码示例，用于在个人笔记本的GeForce RTX 1060上通过主动学习微调轻量级中文文本模型（chinese-roberta-wwm-ext）进行中文文本的多分类任务。
    包含完整的代码、注释和少量训练集样本。

    包含如下功能：
    1、支持LoRa微调
    2、支持QLoRA微调（新增）
    3、支持4位量化（新增）
    4、支持训练过程Train和Validation的Loss和Accuracy可视化输出
    5、基础模型和分类头和LoRa适配器一起合并保存（（具体保存地址在：{output_dir}/deployment_model/））以保证测试时重新正确加载，单独分离保存基础模型（含分类头）无法保证测试时重新正确加载（具体表现是测试加载时候报分类头参数初始化异常，
       而且测试集准确率远低于验证集准确率，原因见下面的"重要补充说明"）
    6、LoRa适配器另外单独保存以用于后续训练（具体保存地址在：{output_dir}/training_adapter/）

    这个代码设计为在NVIDIA GeForce RTX 1060（显存6GB）上高效运行，主动学习过程可以帮助你理解如何逐步改进模型性能，同时保持计算资源在可管理范围内。

    by 李明华，2025-08-16.

    重要补充说明：
    关于分类头：
       - 在训练时，分类头是基础模型的一部分（`BertForSequenceClassification`自带分类头），我们通过`modules_to_save`参数确保分类头也被微调。
       - 合并后，分类头的权重就是微调后的权重，因此在测试时直接使用即可。
    因此，我建议放弃分离保存的方式，采用合并保存。这样简单可靠，且测试集性能与验证集一致。
    如果坚持要分离保存，那么需要确保基础模型以原始结构保存，但实际上PEFT训练后的基础模型已经不再是原始结构。所以分离保存比较复杂，容易出错。

    分离关注点：
        deployment_model：合并后的完整模型，适合部署
        training_adapter：纯适配器，适合恢复训练
        full_state.pth：完整状态（可选），用于调试

    by 李明华，2025-08-18.

    支持label在前、text在后、tab分隔、无表头的输入格式。
    修改 by 李明华，2025-09-01.

    新增QLoRA和4位量化支持（待完善）
    修改 by 李明华，2025-09-06.
'''

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
# 英文分词器
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
# 中文分词器
from transformers import AutoTokenizer, AutoModelForSequenceClassification, BertConfig, AutoConfig
from transformers import AdamW, get_linear_schedule_with_warmup
from peft import LoraConfig, get_peft_model, TaskType
from peft import PeftModel
import numpy as np
from sklearn.metrics import accuracy_score, classification_report

from tqdm import tqdm
import random
import sys
import time

import os
import pandas as pd
from sklearn.model_selection import train_test_split
from collections import defaultdict
import re
import jieba

# 新增评估指标计算函数
from sklearn.metrics import precision_recall_fscore_support, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import font_manager

import json



# 检查GPU可用性
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


# 导入bitsandbytes用于4位量化
try:
    import bitsandbytes as bnb
    from transformers import BitsAndBytesConfig
    # 检查CUDA兼容性
    if torch.cuda.get_device_capability()[0] >= 7:  # Volta及更新架构
        BITSANDBYTES_AVAILABLE = True
        print("bitsandbytes 库可用，支持4位量化")
    else:
        BITSANDBYTES_AVAILABLE = False
        print("当前GPU架构可能不完全支持bitsandbytes 4位量化，使用普通LoRA")
except ImportError:
    BITSANDBYTES_AVAILABLE = False
    print("bitsandbytes 库不可用，将使用普通LoRA")

# 修改 use_qlora 和 use_4bit_quantization 的设置
# 根据GPU架构自动调整
gpu_arch = torch.cuda.get_device_capability()[0]
if gpu_arch >= 7:  # Volta(Turing, Ampere, Ada Lovelace, Hopper)
    # QLoRA配置
    use_qlora = True  # 使用QLoRA
    use_4bit_quantization = True  # 使用4位量化
    print(f"GPU架构支持量化: SM{gpu_arch}.x")
else:
    use_qlora = False
    use_4bit_quantization = False
    print(f"GPU架构可能不完全支持量化(SM{gpu_arch}.x)，使用普通LoRA")


# 设置为离线模式
os.environ['TRANSFORMERS_OFFLINE'] = '1'

# 训练超参数设置
batch_size = 16
num_epochs = 1
learning_rate = 1e-4
weight_decay = 0.01  # L2正则化，防止过拟合
num_workers = 1
num_labels = 10  # 分类数目
max_samples = 5000
num_warmup_steps = 2
# 分类头参数
hidden_dim = 50

# 模型文件配置
model_dir = ('/workspace/MyProjects/tg-demo-slim/text-test/models/')  # 本地模型权重
model_name = 'chinese-roberta-wwm-ext'
train_history_filepath = './training_history-text.png'

# 训练集
train_data_path = './data/preprocess/mini_train_set.csv'
# 测试集
test_data_path = './data/preprocess/mini_test_set.csv'

# LoRa微调模型文件配置
use_lora = True

# 训练过程输出
lora_train_history_filepath = './training_history-text_lora.png'

# 模型结果保存目录
OUTPUT_DIR = './outputs/ch_text_classify_lora_full'  # 注意结尾不要带'/'
os.makedirs(OUTPUT_DIR, exist_ok=True)
HIGH_ENTROPY_SAMPLES_DIR = './outputs/high_entropy_samples'
os.makedirs(OUTPUT_DIR, exist_ok=True)
# 最佳检查点的epoch（初始值默认为0，跑完后会更新）
BEST_CHECKPOINT_EPOCH = 0


# 中文预处理函数
def preprocess_chinese_text(text):
    """中文文本预处理流程"""
    # 1. 去除特殊字符和多余空格
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()

    # 2. 分词处理（可选）
    # text = " ".join(jieba.cut(text))

    return text


# 自定义数据集类
class ChineseMovieReviewDataset(Dataset):
    """中文数据集"""

    def __init__(self, texts, labels, tokenizer, max_length=256):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx]

        # 中文预处理
        text = preprocess_chinese_text(text)

        # 编码文本
        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt',
        )

        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }


# 数据读取函数 - 修改为支持label在前、text在后、tab分隔、无表头
def load_and_split_THUCNews_data(file_path, test_size=0.2, random_state=42):
    """加载THUCNews数据并分割为训练集和验证集"""
    # 创建小型训练子集，保持原始标签分布
    df = create_mini_dataset_fast(file_path, max_samples=max_samples, random_state=42)

    # 先使用中文标签进行统计
    print(f"训练集 {file_path} 的表头： ", df.head())

    # 统计各label对应的样本数（按数量降序排列）- 使用中文标签
    label_counts = df['label'].value_counts()
    print("\n各标签的样本数量统计（中文标签）：")
    print(label_counts)

    # 更详细的统计（包括百分比）- 使用中文标签
    total_samples = len(df)
    label_stats = df['label'].value_counts().reset_index()
    label_stats.columns = ['label_name', 'count']
    label_stats['percentage'] = (label_stats['count'] / total_samples * 100).round(2)
    print("\n详细的标签分布统计（中文标签）：")
    print(label_stats)

    # 创建标签映射字典（将中文标签映射为数字）
    unique_labels = df['label'].unique()
    label_to_id = {label: idx for idx, label in enumerate(unique_labels)}
    id_to_label = {idx: label for label, idx in label_to_id.items()}

    print(f"\n标签映射关系: {label_to_id}")

    # 将中文标签转换为数字标签
    df['label'] = df['label'].map(label_to_id)

    # 划分训练集和验证集
    train_df, val_df = train_test_split(
        df,
        test_size=test_size,
        random_state=random_state,
        stratify=df['label']  # 分层抽样保持分布
    )

    print(f"\n数据集大小: 总共 {len(df)} 条")
    print(f"训练集: {len(train_df)} 条")
    print(f"验证集: {len(val_df)} 条")

    # 统计标签分布 - 使用数字标签但显示中文标签名称
    print("\n训练集标签分布:")
    train_label_counts = train_df['label'].value_counts()
    for label_id, count in train_label_counts.items():
        label_name = id_to_label[label_id]
        print(f"  {label_name} ({label_id}): {count} 条")

    print("\n验证集标签分布:")
    val_label_counts = val_df['label'].value_counts()
    for label_id, count in val_label_counts.items():
        label_name = id_to_label[label_id]
        print(f"  {label_name} ({label_id}): {count} 条")

    return train_df, val_df, label_to_id, id_to_label


def create_mini_dataset_fast(original_file, max_samples=1000, random_state=42):
    """
    高效地在内存中创建小型训练子集，保持原始标签分布
    参数:
        original_file: 原始训练集文件路径
        max_samples: 子集最大样本数
        random_state: 随机种子
    返回:
        mini_df: 包含采样后数据的小型DataFrame
    """
    # 读取原始数据 - 修改为支持label在前、text在后、tab分隔、无表头
    print("正在读取数据...")
    start_time = time.time()
    df = pd.read_csv(original_file, sep='\t', header=None, names=['label', 'text'])
    load_time = time.time() - start_time
    print(f"原始训练集大小：{len(df)} 条，读取耗时：{load_time:.2f}秒")
    print(f"开始分层抽样生成大小为 {max_samples} 条的 mini train set")

    # 使用优化后的分层抽样函数
    start_sample_time = time.time()
    mini_df = optimized_stratified_sample(df, 'label', max_samples, random_state)
    sample_time = time.time() - start_sample_time

    # 打印统计信息
    print(f"\n创建内存中的小型训练子集完成: 共 {len(mini_df)} 条样本，抽样耗时：{sample_time:.2f}秒")
    print("标签分布:")
    print(mini_df['label'].value_counts())
    print(f"抽样前后比例一致性检查:")
    original_prop = df['label'].value_counts(normalize=True).sort_index()
    sampled_prop = mini_df['label'].value_counts(normalize=True).sort_index()
    for label in original_prop.index:
        if label in sampled_prop:
            diff = abs(original_prop[label] - sampled_prop[label])
            print(
                f"  标签 {label}: 原始 {original_prop[label]:.3f} -> 抽样 {sampled_prop[label]:.3f} (差异: {diff:.3f})")

    return mini_df


def optimized_stratified_sample(df, label_col, n_samples, random_state=None):
    """
    高效的分层抽样函数，专门优化用于文本数据
    """
    if n_samples >= len(df):
        return df.copy()

    if random_state is not None:
        np.random.seed(random_state)

    # 1. 预先获取所有标签（向量化操作）
    all_labels = df[label_col].values

    # 2. 使用numpy高效计算类别索引
    unique_classes, class_counts = np.unique(all_labels, return_counts=True)
    class_indices = {}

    # 一次性构建所有类别的索引映射
    for i, class_val in enumerate(unique_classes):
        # 使用numpy的where函数比循环更快
        class_indices[class_val] = np.where(all_labels == class_val)[0]

    # 3. 计算每个类别的抽样数量（保持比例）
    total_count = len(df)
    sample_sizes = {}

    for class_val in unique_classes:
        class_count = len(class_indices[class_val])
        sample_sizes[class_val] = max(1, int(n_samples * class_count / total_count))

    # 4. 调整样本总数
    total_selected = sum(sample_sizes.values())
    if total_selected != n_samples:
        # 按类别大小排序调整
        diff = n_samples - total_selected
        # 按类别大小降序排列
        sorted_classes = sorted([(cls, len(class_indices[cls])) for cls in unique_classes],
                                key=lambda x: x[1], reverse=True)

        for cls, count in sorted_classes:
            if diff == 0:
                break
            current_size = sample_sizes[cls]
            available = count - current_size
            if available > 0:
                to_add = min(diff, available)
                sample_sizes[cls] += to_add
                diff -= to_add

    # 5. 收集抽样索引（使用numpy随机选择）
    sampled_indices = []
    for class_val in unique_classes:
        size = sample_sizes[class_val]
        indices = class_indices[class_val]

        if size > 0:
            if len(indices) > size:
                selected = np.random.choice(indices, size=size, replace=False)
                sampled_indices.extend(selected)
            else:
                sampled_indices.extend(indices)

    # 6. 如果样本不足，补充随机样本（确保总数正确）
    if len(sampled_indices) < n_samples:
        all_indices = np.arange(len(df))
        used_mask = np.isin(all_indices, sampled_indices)
        remaining_indices = all_indices[~used_mask]

        extra_needed = n_samples - len(sampled_indices)
        if len(remaining_indices) > 0:
            extra_selected = np.random.choice(remaining_indices,
                                              size=min(extra_needed, len(remaining_indices)),
                                              replace=False)
            sampled_indices.extend(extra_selected)

    # 7. 创建结果DataFrame并打乱顺序
    result_df = df.iloc[sampled_indices].copy()
    result_df = result_df.sample(frac=1, random_state=random_state).reset_index(drop=True)

    return result_df


def load_THUCNews_test(file_path, label_to_id):
    """加载测试集数据 - 修改为支持label在前、text在后、tab分隔、无表头"""
    # 读取Tab分隔的.txt文件，label在前，text在后，无表头
    test_df = pd.read_csv(file_path, sep='\t', header=None, names=['label', 'text'])

    # 将中文标签转换为数字标签
    test_df['label'] = test_df['label'].map(label_to_id)

    # 检查是否有未映射的标签
    if test_df['label'].isna().any():
        print("警告: 测试集中存在训练时未见过的标签")
        # 可以选择删除这些样本或赋予一个默认值
        test_df = test_df.dropna(subset=['label'])
        test_df['label'] = test_df['label'].astype(int)

    print(f"测试集 {file_path} 的表头： ", test_df.head())
    # 统计各label对应的样本数（按数量降序排列）
    label_counts = test_df['label'].value_counts()
    # 打印结果
    print("\n各标签的样本数量统计：")
    print(label_counts)
    # 更详细的统计（包括百分比）
    total_samples = len(test_df)
    label_stats = test_df['label'].value_counts().reset_index()
    label_stats.columns = ['label_id', 'count']
    label_stats['percentage'] = (label_stats['count'] / total_samples * 100).round(2)
    print("\n详细的标签分布统计：")
    print(label_stats)

    print(f"测试集大小: 总共 {len(test_df)} 条")

    # 统计标签分布
    print("\n测试集标签分布:")
    print(test_df['label'].value_counts())

    return test_df


# 数据处理流程
def prepare_datasets(model_name, train_data_path):
    """完整数据准备流程"""
    # 1. 加载并分割数据
    train_df, val_df, label_to_id, id_to_label = load_and_split_THUCNews_data(train_data_path)

    # 更新全局变量
    global num_labels
    num_labels = len(label_to_id)
    print(f"实际类别数量: {num_labels}")

    print(f"实际使用训练集大小： {len(train_df)} 条")
    print(f"实际使用验证集大小： {len(val_df)} 条")

    # 判断训练集中类别数目与设定数目是否一致
    assert train_df['label'].max() == (num_labels - 1), "标签需转换为0-(num_labels)的连续整数"

    # 2. 初始化分词器
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        model_max_length=256
    )

    # 3. 创建数据集
    train_dataset = ChineseMovieReviewDataset(
        train_df['text'].values,
        train_df['label'].values,
        tokenizer
    )

    val_dataset = ChineseMovieReviewDataset(
        val_df['text'].values,
        val_df['label'].values,
        tokenizer
    )

    # 4. 创建数据加载器
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers
    )

    return train_loader, val_loader, tokenizer, label_to_id, id_to_label


# 方案二：加入384维中间层和ReLU激活函数提升非线性能力
class EnhancedClassifier(nn.Module):
    def __init__(self, hidden_size=768, hidden_dim=384, num_labels=10):
        super().__init__()
        self.dense = nn.Linear(hidden_size, hidden_dim)
        self.dropout = nn.Dropout(0.1)
        self.output = nn.Linear(hidden_dim, num_labels)

    def forward(self, x):
        x = self.dense(x)
        x = torch.relu(x)  # 或GELU
        x = self.dropout(x)
        return self.output(x)


# 方案三：更深的分类头
class DeepClassifier(nn.Module):
    def __init__(self, hidden_size=768, num_labels=10):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(hidden_size, 512),
            nn.GELU(),
            nn.LayerNorm(512),
            nn.Linear(512, 256),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(256, num_labels)
        )

    def forward(self, x):
        return self.layers(x)


def create_quantization_config():
    """创建4位量化配置"""
    if not BITSANDBYTES_AVAILABLE or not use_4bit_quantization:
        return None

    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
    )
    print("使用4位量化配置")
    return quantization_config


def create_lora_model(num_labels, pretrained_path, lora_config=None):
    """创建LoRA微调模型（支持QLoRA和普通LoRA）"""
    # 1. 加载配置并添加分类头信息
    quantization_config = create_quantization_config() if use_qlora else None

    config = AutoConfig.from_pretrained(
        pretrained_path,
        local_files_only=True,
        num_labels=num_labels
    )

    # 添加自定义分类头配置
    config.update({
        "classifier_type": "EnhancedClassifier",
        "classifier_hidden_dim": 384,
        "classifier_activation": "GELU"
    })

    # 2. 加载基础模型（支持量化）
    # 移除 device_map="auto"，手动将模型移动到设备
    model = AutoModelForSequenceClassification.from_pretrained(
        pretrained_path,
        config=config,
        local_files_only=True,
        ignore_mismatched_sizes=True,
        quantization_config=quantization_config if use_qlora else None,
        # device_map="auto" if use_qlora else None,
        torch_dtype=torch.float16 if use_qlora else torch.float32
    )

    # 手动将模型移动到设备
    model = model.to(device)

    # 3. 配置LoRA参数（增强版配置）
    if lora_config is None:
        lora_config = LoraConfig(
            task_type=TaskType.SEQ_CLS,
            r=8 if use_qlora else 16,  # QLoRA使用较小的秩
            lora_alpha=16 if use_qlora else 32,
            target_modules=["query", "key", "value", "dense"],
            lora_dropout=0.1,
            modules_to_save=["classifier"],
            bias="none"
        )

    # 4. 转换为LoRA模型
    lora_model = get_peft_model(model, lora_config)

    # 5. 打印可训练参数和量化信息
    total_params = sum(p.numel() for p in lora_model.parameters())
    trainable_params = sum(p.numel() for p in lora_model.parameters() if p.requires_grad)

    print("\n" + "=" * 50)
    print(f"模型类型: {'QLoRA (4位量化)' if use_qlora else '普通LoRA'}")
    print(f"总参数数量: {total_params:,}")
    print(f"可训练参数数量: {trainable_params:,} (占 {trainable_params / total_params * 100:.2f}%)")
    print(f"内存节省比例: {(1 - trainable_params / total_params) * 100:.1f}%")
    print("=" * 50 + "\n")

    return lora_model.to(device)


def evaluate_model(model, data_loader, criterion):
    """评估模型性能"""
    model.eval()
    running_loss = 0.0
    running_corrects = 0

    with torch.no_grad():
        for batch in tqdm(data_loader, desc="Validating"):
            # 处理返回三个值的情况（图片ID）
            if len(batch) == 3:
                inputs, labels, _ = batch[0], batch[1], batch[2]
            else:
                inputs, labels = batch[0], batch[1]

            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            loss = criterion(outputs, labels)
            _, preds = torch.max(outputs, 1)

            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)

    loss = running_loss / len(data_loader.dataset)
    acc = running_corrects.double() / len(data_loader.dataset)

    return loss, acc


# 5. 训练循环
def train_epoch(model, data_loader, optimizer, scheduler, device):
    model.train()
    losses = []
    correct_predictions = 0

    for batch in tqdm(data_loader, desc="Training"):
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)  # 数据读入到GPU

        # 前向传播
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels
        )

        loss = outputs.loss
        logits = outputs.logits

        # 计算准确率
        _, preds = torch.max(logits, dim=1)
        correct_predictions += torch.sum(preds == labels)

        losses.append(loss.item())

        # 反向传播
        loss.backward()

        # 梯度裁剪防止爆炸
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        # 参数更新
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()

    # 计算平均损失和准确率
    avg_loss = np.mean(losses)
    accuracy = correct_predictions.double() / len(data_loader.dataset)

    return avg_loss, accuracy


def train_and_evaluate_lora(model, train_loader, val_loader, output_dir, num_epochs=3, learning_rate=1e-4):
    """LoRA专用训练函数"""
    # 训练前打印显存信息
    initial_mem = torch.cuda.memory_allocated() / 1024 ** 2
    print(f"训练前显存占用: {initial_mem:.2f} MB")
    print(f"模型类型: {'QLoRA (4位量化)' if use_qlora else '普通LoRA'}")

    # 优化器仅更新可训练参数
    optimizer = AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=learning_rate * (2 if use_qlora else 1),  # QLoRA通常需要更大学习率
        weight_decay=weight_decay
    )
    EPOCHS = num_epochs
    total_steps = len(train_loader) * EPOCHS
    print('训练步数: ', total_steps)
    # 学习率调度
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=total_steps
    )

    # 记录训练过程
    history = {
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': []
    }
    best_val_acc = 0
    best_checkpoint_epoch = 0

    # 记录训练时间
    training_times = []

    for epoch in range(EPOCHS):
        print(f"\nEpoch {epoch + 1}/{EPOCHS}")
        print("-" * 10)

        # 训练
        epoch_start_time = time.time()
        train_loss, train_acc = train_epoch(model, train_loader, optimizer, scheduler, device)
        epoch_time = time.time() - epoch_start_time
        training_times.append(epoch_time)

        history['train_loss'].append(float(train_loss))
        history['train_acc'].append(float(train_acc))
        print(f"Train loss: {train_loss:.4f}, accuracy: {train_acc:.4f}")
        print(f"Epoch 耗时: {epoch_time:.2f}秒")

        # 评估
        val_loss, val_acc = eval_model(model, val_loader, device)
        history['val_loss'].append(float(val_loss))
        history['val_acc'].append(float(val_acc))
        print(f"Validation loss: {val_loss:.4f}, accuracy: {val_acc:.4f}")

        train_classifier_norm = torch.norm(model.classifier.weight).item()
        print(f"训练结束时分类头权重范数: {train_classifier_norm:.4f} , 应该和测试前加载的权重一样")

        # 学习率调整
        scheduler.step(val_acc)

        # 显存监控
        epoch_mem = torch.cuda.max_memory_allocated() / 1024 ** 2
        print(f"Epoch {epoch + 1} 最大显存占用: {epoch_mem:.2f} MB")
        torch.cuda.reset_peak_memory_stats()

        # 保存最佳检查点
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_checkpoint_epoch = epoch
            # 同时保存部署模型和训练适配器
            save_best_checkpoint(model, output_dir)

            # 保存优化器状态（用于恢复训练）
            torch.save(optimizer.state_dict(), f"{output_dir}/training_adapter/optimizer.pth")
            print(f"--> 提醒： 保存最佳检查点: epoch = {epoch}, 路径为 {output_dir}/training_adapter/ ")

    # 计算平均训练时间
    avg_training_time = np.mean(training_times)
    print(f"\n平均每个epoch训练时间: {avg_training_time:.2f}秒")
    print(f"总训练时间: {sum(training_times):.2f}秒")

    BEST_CHECKPOINT_EPOCH = best_checkpoint_epoch
    print("---")
    print(f"完成模型训练！")
    print(
        f"最佳检查点的epoch数为 best_checkpoint_epoch : {BEST_CHECKPOINT_EPOCH}，最佳检查点保存路径为 {output_dir}/training_adapter/ ")

    # === 训练完成后再合并保存 ===
    print("\n训练完成，合并最佳模型...")
    merged_model = save_merged_model(model, output_dir=output_dir)

    return best_val_acc, history, avg_training_time


def save_merged_model(model, output_dir):
    """合并LoRA权重后保存完整模型"""
    # 合并LoRA权重到基础模型
    if use_qlora:
        print("QLoRA模型需要先解除量化再合并...")
        # QLoRA需要特殊处理
        from peft import prepare_model_for_kbit_training
        model = prepare_model_for_kbit_training(model)

    merged_model = model.merge_and_unload()

    # 保存完整模型
    deployment_dir = os.path.join(output_dir, "deployment_model")
    merged_model.save_pretrained(deployment_dir)
    tokenizer.save_pretrained(deployment_dir)

    # 保存标签映射信息到config.json
    config_path = os.path.join(deployment_dir, "config.json")
    if os.path.exists(config_path):
        with open(config_path, 'r', encoding='utf-8') as f:
            config = json.load(f)

        # 添加标签映射信息
        config['id2label'] = id_to_label
        config['label2id'] = label_to_id
        # config['model_type'] = 'qlora' if use_qlora else 'lora'
        config['quantization'] = use_4bit_quantization

        with open(config_path, 'w', encoding='utf-8') as f:
            json.dump(config, f, ensure_ascii=False, indent=4)

    print(f"部署模型已保存到 {deployment_dir}")
    print(f"合并后的完整模型已保存到 {output_dir}")

    # 验证保存效果
    print("合并后模型分类头权重范数:", torch.norm(merged_model.classifier.weight).item())

    return merged_model


def save_best_checkpoint(model, output_dir):
    """同时保存最佳检查点的适配器（用于迭代训练）"""

    # 2. 保存适配器（用于迭代训练）
    # 注意：这里保存的是未合并的适配器
    model.save_pretrained(os.path.join(output_dir, "training_adapter"))
    print(f"训练适配器已保存到 {output_dir}/training_adapter")

    # 3. 额外保存完整状态（可选）
    torch.save(model.state_dict(), os.path.join(output_dir, "full_state.pth"))
    print(f"完整状态已保存到 {output_dir}/full_state.pth")

    return model


def resume_training(adapter_path, base_model_path, num_labels):
    """从保存的适配器恢复训练"""
    # 1. 加载基础模型
    config = AutoConfig.from_pretrained(
        base_model_path,
        num_labels=num_labels
    )
    base_model = AutoModelForSequenceClassification.from_pretrained(
        base_model_path,
        config=config
    )

    # 2. 加载适配器
    model = PeftModel.from_pretrained(
        base_model,
        adapter_path,
        is_trainable=True  # 设置为可训练
    )

    # 3. 恢复优化器状态（如果有保存）
    optimizer_path = os.path.join(adapter_path, "optimizer.pth")
    if os.path.exists(optimizer_path):
        optimizer.load_state_dict(torch.load(optimizer_path))
        print("优化器状态已恢复")

    return model


# 训练过程可视化输出
def plot_training_history(history):
    plt.figure(figsize=(12, 4))

    # 损失曲线
    plt.subplot(1, 2, 1)
    plt.plot(history['train_loss'], label='Train Loss')
    plt.plot(history['val_loss'], label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    # 准确率曲线
    plt.subplot(1, 2, 2)
    plt.plot(history['train_acc'], label='Train Accuracy')
    plt.plot(history['val_acc'], label='Validation Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.tight_layout()
    plt.savefig(lora_train_history_filepath)
    plt.close()


# 6. (训练时的）评估函数
def eval_model(model, data_loader, device):
    model.eval()
    losses = []
    correct_predictions = 0

    with torch.no_grad():
        for batch in tqdm(data_loader, desc="Validation"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            # 前向传播
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )

            loss = outputs.loss
            logits = outputs.logits

            # 计算准确率
            _, preds = torch.max(logits, dim=1)
            correct_predictions += torch.sum(preds == labels)
            losses.append(loss.item())

    # 计算平均损失和准确率
    avg_loss = np.mean(losses)
    accuracy = correct_predictions.double() / len(data_loader.dataset)

    return avg_loss, accuracy


# 11. 测试模型
def predict_category(model, tokenizer, text, device, id_to_label):
    """预测新闻类别（适配THUCNews的10分类）"""
    model.eval()
    inputs = tokenizer(
        text,
        add_special_tokens=True,
        max_length=128,
        padding='max_length',
        truncation=True,
        return_tensors='pt'
    ).to(device)

    with torch.no_grad():
        outputs = model(**inputs)
        probs = torch.nn.functional.softmax(outputs.logits, dim=1)
        pred = torch.argmax(probs, dim=1).item()

    return id_to_label[pred], probs.cpu().numpy()[0]


# ==================== 4. 测试集评估 ====================
def load_test_data(file_path):
    """加载测试集数据 - 修改为支持label在前、text在后、tab分隔、无表头"""
    test_df = pd.read_csv(file_path, sep='\t', header=None, names=['label', 'text'])
    print(f"\n测试集 {file_path} 的统计信息：")
    print(f"总样本数: {len(test_df)}")
    print("标签分布:")
    print(test_df['label'].value_counts())
    return test_df


# 包含最大熵样本统计和输出的测试函数
def test_model_lora(model, test_loader, device, test_df, high_entropy_samples_dir, id_to_label):
    """
    增强版测试评估函数
    功能：
    1. 计算标准测试指标
    2. 识别预测不确定性高的样本（最大熵）
    3. 记录分类错误的样本信息到文件
    """
    # 创建输出目录
    os.makedirs(high_entropy_samples_dir, exist_ok=True)
    error_log_file = os.path.join(high_entropy_samples_dir, 'error_samples.txt')
    entropy_log_file = os.path.join(high_entropy_samples_dir, 'high_entropy_samples.txt')

    model.eval()
    total_correct = 0
    total_samples = 0
    test_loss = 0
    all_preds = []
    all_labels = []

    # 记录推理时间
    inference_start_time = time.time()

    # 新增数据结构
    error_samples = []  # 存储错误样本信息
    high_entropy_samples = []  # 存储高熵样本信息
    text_list = test_df['text'].tolist()  # 假设test_df包含原始文本

    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(test_loader, desc="Testing")):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )

            loss = outputs.loss
            logits = outputs.logits

            # 计算概率和预测结果
            probs = torch.nn.functional.softmax(logits, dim=1)
            entropy = -torch.sum(probs * torch.log(probs + 1e-10), dim=1)  # 计算熵
            _, preds = torch.max(logits, dim=1)

            # 统计指标
            total_correct += (preds == labels).sum().item()
            total_samples += labels.size(0)
            test_loss += loss.item()

            # 收集预测结果
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

            # 记录错误样本和高熵样本
            batch_entropy = entropy.cpu().numpy()
            batch_texts = text_list[batch_idx * batch_size: (batch_idx + 1) * batch_size]

            for i in range(len(labels)):
                sample_idx = batch_idx * batch_size + i
                sample_info = {
                    'text': batch_texts[i],
                    'true_label': labels[i].item(),
                    'true_label_name': id_to_label[labels[i].item()],
                    'pred_label': preds[i].item(),
                    'pred_label_name': id_to_label[preds[i].item()],
                    'probabilities': probs[i].cpu().numpy(),
                    'entropy': batch_entropy[i]
                }

                # 记录错误样本
                if preds[i] != labels[i]:
                    error_samples.append(sample_info)

                # 记录高熵样本（熵大于平均熵+标准差）
                if batch_entropy[i] > np.mean(batch_entropy) + np.std(batch_entropy):
                    high_entropy_samples.append(sample_info)

    # 计算推理时间
    inference_time = time.time() - inference_start_time
    avg_inference_time_per_sample = inference_time / total_samples

    # 计算评估指标
    avg_loss = test_loss / len(test_loader)
    accuracy = total_correct / total_samples

    # 打印基础结果
    print(f"\n测试集结果:")
    print(f"模型类型: {'QLoRA (4位量化)' if use_qlora else '普通LoRA'}")
    print(f"平均损失: {avg_loss:.4f}")
    print(f"准确率: {accuracy:.4f}")
    print(f"错误样本数: {len(error_samples)}/{total_samples}")
    print(f"高不确定性样本数: {len(high_entropy_samples)}/{total_samples}")
    print(f"总推理时间: {inference_time:.2f}秒")
    print(f"平均每个样本推理时间: {avg_inference_time_per_sample * 1000:.2f}毫秒")

    # 分类报告（使用中文标签名称）
    target_names = [id_to_label[i] for i in sorted(id_to_label.keys())]
    print("\n分类报告:")
    print(classification_report(all_labels, all_preds,
                                target_names=target_names))

    # 保存错误样本到文件
    with open(error_log_file, 'w', encoding='utf-8') as f:
        f.write("=== 分类错误样本 ===\n")
        f.write(f"总数: {len(error_samples)}\n\n")
        for sample in sorted(error_samples, key=lambda x: x['entropy'], reverse=True)[:100]:  # 最多保存100个
            f.write(f"文本: {sample['text']}\n")
            f.write(f"真实标签: {sample['true_label_name']} ({sample['true_label']})\n")
            f.write(f"预测标签: {sample['pred_label_name']} ({sample['pred_label']})\n")
            f.write(f"各类别概率: {np.round(sample['probabilities'], 4)}\n")
            f.write(f"熵值: {sample['entropy']:.4f}\n")
            f.write("-" * 50 + "\n")

    # 保存高熵样本到文件
    with open(entropy_log_file, 'w', encoding='utf-8') as f:
        f.write("=== 高不确定性样本 ===\n")
        f.write(f"总数: {len(high_entropy_samples)}\n\n")
        for sample in sorted(high_entropy_samples, key=lambda x: x['entropy'], reverse=True)[:100]:
            f.write(f"文本: {sample['text']}\n")
            f.write(f"真实标签: {sample['true_label_name']} ({sample['true_label']})\n")
            f.write(f"预测标签: {sample['pred_label_name']} ({sample['pred_label']})\n")
            f.write(f"各类别概率: {np.round(sample['probabilities'], 4)}\n")
            f.write(f"熵值: {sample['entropy']:.4f}\n")
            f.write("-" * 50 + "\n")

    # 打印部分样本示例（使用中文标签）
    print("\n=== 分类错误样本示例 ===")
    for sample in error_samples[:3]:
        print(f"文本: {sample['text'][:50]}...")
        print(f"真实标签: {sample['true_label_name']} | 预测: {sample['pred_label_name']}")
        print(f"熵值: {sample['entropy']:.4f}\n")

    print("\n=== 高不确定性样本示例 ===")
    for sample in high_entropy_samples[:3]:
        print(f"文本: {sample['text'][:50]}...")
        print(f"预测标签: {sample['pred_label_name']} | 熵值: {sample['entropy']:.4f}")
        print(f"概率分布: {np.round(sample['probabilities'], 2)}\n")

    return {
        'accuracy': accuracy,
        'loss': avg_loss,
        'error_samples': error_samples,
        'high_entropy_samples': high_entropy_samples,
        'inference_time': inference_time,
        'avg_inference_time_per_sample': avg_inference_time_per_sample
    }


# ==================== 5. 单条样本测试示例 ====================
def predict_example(model, tokenizer, text, device, id_to_label):
    """预测单条样本"""
    model.eval()
    inputs = tokenizer(
        text,
        add_special_tokens=True,
        max_length=256,
        padding='max_length',
        truncation=True,
        return_tensors='pt'
    ).to(device)

    # 记录推理时间
    start_time = time.time()

    with torch.no_grad():
        outputs = model(**inputs)
        probs = torch.nn.functional.softmax(outputs.logits, dim=1)
        pred = torch.argmax(probs, dim=1).item()

    inference_time = time.time() - start_time

    return id_to_label[pred], probs.cpu().numpy()[0], inference_time


def load_trained_lora_merged_model(output_dir):
    """加载训练好的、合并后的完整模型"""
    deployment_model_path = f"{output_dir}/deployment_model"
    if not os.path.exists(deployment_model_path):
        raise FileNotFoundError(f"lora模型文件 {deployment_model_path} 不存在！")

    # 从config.json读取模型类型信息
    config_path = os.path.join(deployment_model_path, "config.json")
    model_type = 'bert'  # 默认使用bert类型
    quantization = False
    peft_type = 'lora'

    if os.path.exists(config_path):
        with open(config_path, 'r', encoding='utf-8') as f:
            config = json.load(f)

        # 读取自定义字段，但加载时使用原始model_type
        peft_type = config.get('peft_type', 'lora')
        quantization = config.get('quantization', False)

        # 确保使用原始的model_type
        if 'model_type' in config and config['model_type'] not in ['qlora', 'lora']:
            model_type = config['model_type']
        else:
            # 回退到默认类型
            model_type = 'bert'

    print(f"加载模型 - 基础类型: {model_type}, PEFT类型: {peft_type}, 量化: {quantization}")

    # 测试时使用部署模型
    merged_model = AutoModelForSequenceClassification.from_pretrained(deployment_model_path)

    # 从config.json加载标签映射信息
    if os.path.exists(config_path):
        with open(config_path, 'r', encoding='utf-8') as f:
            config = json.load(f)

        # 修复：将字符串键转换为整数键
        id2label = config.get('id2label', {})
        label2id = config.get('label2id', {})

        # 转换id2label的键为整数
        id_to_label = {int(k): v for k, v in id2label.items()}
        label_to_id = {k: int(v) for k, v in label2id.items()}

        print(f"从配置文件加载标签映射: {id_to_label}")

        # 设置为全局变量
        globals()['id_to_label'] = id_to_label
        globals()['label_to_id'] = label_to_id

    loaded_classifier_norm = torch.norm(merged_model.classifier.weight).item()
    print("\n=== 合并模型加载验证 ===")
    print(f"加载后分类头权重范数: {loaded_classifier_norm:.4f}, 应该和训练结束时候的一样")

    return merged_model.to(device)


def verify_model_loading(model, test_text, id_to_label):
    """验证模型是否正确加载"""
    pred, probs, inference_time = predict_example(model, tokenizer, test_text, device, id_to_label)
    print(f"预测结果: {pred}")
    print(f"单条样本推理时间: {inference_time * 1000:.2f}毫秒")

    # 检查分类头权重
    classifier_weight = model.classifier.weight.data
    print(f"分类头权重范数: {torch.norm(classifier_weight).item():.4f}")

    # 检查LoRA层是否存在
    has_lora = any('lora' in name for name, _ in model.named_parameters())
    print(f"包含LoRA层: {has_lora}")


def check_weight_consistency(model1, model2):
    """检查两个模型权重是否一致"""
    for (name1, param1), (name2, param2) in zip(model1.named_parameters(), model2.named_parameters()):
        if name1 != name2:
            print(f"参数名不匹配: {name1} vs {name2}")
            return False

        diff = torch.norm(param1.data - param2.data).item()
        if diff > 1e-6:
            print(f"权重不一致: {name1}, 差异={diff:.6f}")
            return False

    print("所有权重一致!")
    return True


def compare_performance(standard_results, qlora_results):
    """对比标准LoRA和QLoRA的性能"""
    print("\n" + "=" * 60)
    print("性能对比: 标准LoRA vs QLoRA (4位量化)")
    print("=" * 60)

    print(f"{'指标':<25} {'标准LoRA':<15} {'QLoRA':<15} {'变化':<10}")
    print("-" * 60)

    # 准确率对比
    acc_diff = qlora_results['accuracy'] - standard_results['accuracy']
    print(f"{'准确率':<25} {standard_results['accuracy']:.4f}<15 {qlora_results['accuracy']:.4f}<15 {acc_diff:+.4f}")

    # 推理时间对比
    time_std = standard_results['avg_inference_time_per_sample'] * 1000
    time_qlora = qlora_results['avg_inference_time_per_sample'] * 1000
    time_diff = time_qlora - time_std
    print(f"{'推理时间(ms/样本)':<25} {time_std:.2f}<15 {time_qlora:.2f}<15 {time_diff:+.2f}")

    # 显存占用对比（需要从训练日志中获取）
    print(f"{'模型大小':<25} {'标准':<15} {'量化':<15} {'减少':<10}")
    print(f"{'参数数量':<25} {'~110M':<15} {'~28M':<15} {'~75%':<10}")

    print("=" * 60)


# 主程序入口
if __name__ == "__main__":
    # 记录开始时间
    start_time = time.time()

    # === 路线1： 使用预训练模型进行LoRA微调（SFT） ===
    # 1. 创建模型
    print("\n创建LoRA模型...")
    # 初始化LoRA模型（替换原create_model）
    model = create_lora_model(
        num_labels=10,
        pretrained_path=os.path.join(model_dir, model_name),
        lora_config=LoraConfig(
            r=8,
            target_modules=["query", "value"]  # RTX 1060显存有限，减少目标模块
        )
    )
    print("完成LoRa模型创建！")

    # 4. 加载数据，划分训练集和验证集，初始化分词器
    train_loader, val_loader, tokenizer, label_to_id, id_to_label = prepare_datasets(
        os.path.join(model_dir, model_name), train_data_path)
    # 打印一个批次数据
    print("\n训练集批次示例:")
    batch = next(iter(train_loader))
    print(f"输入ID形状: {batch['input_ids'].shape}")
    print(f"注意力掩码形状: {batch['attention_mask'].shape}")
    print(f"标签形状: {batch['labels'].shape}")
    # 解码第一条数据
    sample_text = tokenizer.decode(batch['input_ids'][0])
    print(f"\n第一条样本文本: {sample_text}")
    print(f"对应标签: {batch['labels'][0].item()}")
    # 保存标签映射信息（用于后续测试和部署）
    with open(os.path.join(OUTPUT_DIR, 'label_mapping.json'), 'w', encoding='utf-8') as f:
        json.dump({
            'label_to_id': label_to_id,
            'id_to_label': id_to_label
        }, f, ensure_ascii=False, indent=4)

    # 5. 设置优化器和学习率调度器
    # 6. 执行训练和评估，并自动保存最佳checkpoint
    print("\n开始训练...")
    best_val_acc, history, avg_training_time = train_and_evaluate_lora(
        model,
        train_loader,
        val_loader,
        output_dir=OUTPUT_DIR,
        num_epochs=num_epochs,
        learning_rate=learning_rate
    )

    # 训练后立即验证
    print("\n=== 训练后即时验证 ===")
    val_loss, val_acc = eval_model(model, val_loader, device)
    print(f"验证集结果 - Loss: {val_loss:.4f}, Acc: {val_acc:.4f}")

    # 在主程序训练后调用，训练结果可视化输出（可选）
    plot_training_history(history)

    end_time1 = time.time()
    elapsed_time1 = end_time1 - start_time
    print(f"训练耗时: {elapsed_time1} secs!")
    print(f"平均每个epoch训练时间: {avg_training_time:.2f}秒")

    # --------------- 测试脚本（假设模型已训练并保存为'best_text_model.bin'） ----------------

    # ==================== 1. 加载测试集 ====================
    print("\n加载测试集...")
    # 加载测试集（假设路径为'./data/THUCNews-mini/test.txt'）
    test_df = load_THUCNews_test(test_data_path, label_to_id)

    # ==================== 2. 创建测试数据集 ====================
    test_dataset = ChineseMovieReviewDataset(
        test_df['text'].values,
        test_df['label'].values,
        tokenizer  # 使用和训练时相同的tokenizer
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,  # 和训练时保持一致
        shuffle=False,  # 测试集不需要shuffle
        num_workers=num_workers
    )

    # ==================== 3. 重新加载最佳模型 ====================
    # 重新加载最佳模型进行测试
    # 初始化模型结构（必须和训练时完全一致）
    print("\n重新加载训练好的模型进行测试...")
    test_model = load_trained_lora_merged_model(OUTPUT_DIR)

    # 测试是否正确加载了模型
    # 使用示例
    test_text = "苹果发布新款iPhone"
    verify_model_loading(test_model, test_text, id_to_label)

    # 执行测试
    test_results = test_model_lora(test_model, test_loader, device, test_df,
                                   high_entropy_samples_dir=HIGH_ENTROPY_SAMPLES_DIR, id_to_label=id_to_label)

    # === 测试几条样本
    test_samples = [
        "国足再次失利，球迷表示失望",  # 示例1
        "央行宣布降准0.5个百分点",  # 示例2
        "新款iPhone发布，起售价5999元"  # 示例3
    ]
    print("\n单条样本预测示例:")
    for sample in test_samples:
        pred_label, pred_probs, inference_time = predict_example(test_model, tokenizer, sample, device, id_to_label)
        print(f"文本: {sample}")
        print(f"预测标签: {pred_label}")
        print(f"各类别概率: {pred_probs.round(4)}")
        print(f"推理时间: {inference_time * 1000:.2f}毫秒")
        print("-" * 50)

    # 保存性能结果用于对比
    performance_results = {
        'model_type': 'qlora' if use_qlora else 'lora',
        'accuracy': test_results['accuracy'],
        'inference_time': test_results['inference_time'],
        'avg_inference_time_per_sample': test_results['avg_inference_time_per_sample'],
        'memory_savings': '~75%' if use_qlora else '0%'
    }

    with open(os.path.join(OUTPUT_DIR, 'performance_results.json'), 'w') as f:
        json.dump(performance_results, f, indent=4)

    print('\n---')
    total_time = time.time() - start_time
    test_time = time.time() - elapsed_time1
    print(f"训练耗时: {elapsed_time1} secs!")
    print(f"测试耗时: {test_time} secs!")
    print(f"总耗时: {total_time} secs!")

    # 打印性能总结
    print(f"\n=== 性能总结 ===")
    print(f"模型类型: {'QLoRA (4位量化)' if use_qlora else '标准LoRA'}")
    print(f"测试准确率: {test_results['accuracy']:.4f}")
    print(f"平均推理时间: {test_results['avg_inference_time_per_sample'] * 1000:.2f}毫秒/样本")
    print(f"显存节省: {'~75%' if use_qlora else '无'}")
    print(f"模型大小: {'~28MB' if use_qlora else '~110MB'}")


