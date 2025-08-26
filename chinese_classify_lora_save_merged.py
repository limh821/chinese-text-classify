'''
    以下是一个完整的代码示例，用于在个人笔记本的GeForce RTX 1060上通过主动学习微调轻量级中文文本模型（chinese-roberta-wwm-ext）进行中文文本的多分类任务。
    包含完整的代码、注释和少量训练集样本。

    包含如下功能：
    1、支持LoRa微调
    2、支持训练过程Train和Validation的Loss和Accuracy可视化输出
    3、模型合并保存（没有单独保存分类器，如果不是合并加载，测试集上的准确率会非常低，因为分类头没有单独保存）

    这个代码设计为在NVIDIA GeForce RTX 1060（显存6GB）上高效运行，主动学习过程可以帮助你理解如何逐步改进模型性能，同时保持计算资源在可管理范围内。

    by 李明华，2025-08-18.
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
from sklearn.metrics import accuracy_score
from tqdm import tqdm
import random
import sys
import time

import os
import pandas as pd
from sklearn.model_selection import train_test_split
import re
import jieba

# 新增评估指标计算函数
from sklearn.metrics import precision_recall_fscore_support, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import font_manager





# 检查GPU可用性
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


# 设置为离线模式
os.environ['TRANSFORMERS_OFFLINE'] = '1'

# 训练超参数设置
batch_size = 16
num_epochs = 2
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
# 模型输出
output_model_path = './outputs/ch_text_classify'
ouput_dir = './test_results'
# 测试集
test_data_path = './data/THUCNews-mini/test.txt'

# LoRa微调模型文件配置
use_lora = True
# 模型输出
lora_train_history_filepath = './training_history-text_lora.png'
output_lora_model_path = './outputs/ch_text_classify_lora'


# 在训练前确保输出目录存在
os.makedirs(output_lora_model_path, exist_ok=True)



# 记录开始时间
start_time = time.time()

# 1. 创建小型训练数据集 - 电影评论情感分析（3分类）
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


# 加载数据
def load_and_split_data(file_path, test_size=0.2, random_state=42):
    """加载CSV数据并分割为训练集和验证集"""
    df = pd.read_csv(file_path)

    # 划分训练集和验证集
    train_df, val_df = train_test_split(
        df,
        test_size=test_size,
        random_state=random_state,
        stratify=df['label']  # 分层抽样保持分布
    )

    print(f"数据集大小: 总共 {len(df)} 条")
    print(f"训练集: {len(train_df)} 条")
    print(f"验证集: {len(val_df)} 条")

    # 统计标签分布
    print("\n训练集标签分布:")
    print(train_df['label'].value_counts())
    print("\n验证集标签分布:")
    print(val_df['label'].value_counts())

    return train_df, val_df


def create_mini_dataset_in_memory(original_file, max_samples=1000, random_state=42):
    """
    在内存中创建小型训练子集，保持原始标签分布
    参数:
        original_file: 原始训练集文件路径
        max_samples: 子集最大样本数
        random_state: 随机种子
    返回:
        mini_df: 包含采样后数据的小型DataFrame
    """
    # 读取原始数据
    df = pd.read_csv(original_file, sep='\t', header=None, names=['text', 'label'])
    print(f"原始训练集大小： {len(df)} 条")
    print(f"均匀采样生成大小为 {max_samples} 条的 mini train set ")

    # 计算每个类别应抽取的样本数（按原始比例）
    label_counts = df['label'].value_counts(normalize=True)
    samples_per_label = (label_counts * max_samples).round().astype(int)

    # 确保总数不超过max_samples
    while samples_per_label.sum() > max_samples:
        max_label = samples_per_label.idxmax()
        samples_per_label[max_label] -= 1

    # 分层抽样
    mini_samples = []
    for label, count in samples_per_label.items():
        label_samples = df[df['label'] == label].sample(count, random_state=random_state)
        mini_samples.append(label_samples)

    # 合并并打乱顺序
    mini_df = pd.concat(mini_samples).sample(frac=1, random_state=random_state)

    # 打印统计信息
    print(f"\n创建内存中的小型训练子集: 共 {len(mini_df)} 条样本")
    print("标签分布:")
    print(mini_df['label'].value_counts())

    return mini_df

def load_and_split_THUCNews_data(file_path, test_size=0.2, random_state=42):
    """加载CSV数据并分割为训练集和验证集"""
    # 读取Tab分隔的.txt文件
    # df = pd.read_csv(file_path, sep='\t', header=None, names=['text', 'label'])
    # 创建小型训练子集，保持原始标签分布
    df = create_mini_dataset_in_memory(file_path, max_samples=max_samples, random_state=42)

    print(f"训练集 {file_path} 的表头： ", df.head())
    # 统计各label对应的样本数（按数量降序排列）
    label_counts = df['label'].value_counts()
    # 打印结果
    print("\n各标签的样本数量统计：")
    print(label_counts)
    # 更详细的统计（包括百分比）
    total_samples = len(df)
    label_stats = df['label'].value_counts().reset_index()
    label_stats.columns = ['label', 'count']
    label_stats['percentage'] = (label_stats['count'] / total_samples * 100).round(2)
    print("\n详细的标签分布统计：")
    print(label_stats)

    # 划分训练集和验证集
    train_df, val_df = train_test_split(
        df,
        test_size=test_size,
        random_state=random_state,
        stratify=df['label']  # 分层抽样保持分布
    )

    print(f"数据集大小: 总共 {len(df)} 条")
    print(f"训练集: {len(train_df)} 条")
    print(f"验证集: {len(val_df)} 条")

    # 统计标签分布
    print("\n训练集标签分布:")
    print(train_df['label'].value_counts())
    print("\n验证集标签分布:")
    print(val_df['label'].value_counts())

    return train_df, val_df


def load_THUCNews_test(file_path):
    """ 加载CSV数据并分割为训练集和验证集 """
    # 读取Tab分隔的.txt文件
    test_df = pd.read_csv(file_path, sep='\t', header=None, names=['text', 'label'])

    print(f"测试集 {file_path} 的表头： ", test_df.head())
    # 统计各label对应的样本数（按数量降序排列）
    label_counts = test_df['label'].value_counts()
    # 打印结果
    print("\n各标签的样本数量统计：")
    print(label_counts)
    # 更详细的统计（包括百分比）
    total_samples = len(test_df)
    label_stats = test_df['label'].value_counts().reset_index()
    label_stats.columns = ['label', 'count']
    label_stats['percentage'] = (label_stats['count'] / total_samples * 100).round(2)
    print("\n详细的标签分布统计：")
    print(label_stats)

    print(f"测试集大小: 总共 {len(test_df)} 条")

    # 统计标签分布
    print("\n测试集标签分布:")
    print(test_df['label'].value_counts())

    return test_df


# 数据处理流程
def prepare_datasets(model_name='./models/chinese-roberta-wwm-ext'):
    """完整数据准备流程"""
    # 1. 加载并分割数据
    # train_df, val_df = load_and_split_data('./data/movie_reviews.csv')
    # 针对THUCNew-mini数据集的加载
    train_df, val_df = load_and_split_THUCNews_data('./data/THUCNews-mini/train.txt')
    print(f"实际使用训练集大小： {len(train_df)} 条")
    print(f"实际使用验证集大小： {len(val_df)} 条")
    # 判断训练集中类别数目与设定数目是否一致
    assert train_df['label'].max() == (num_labels-1), "标签需转换为0-(num_labels)的连续整数"

    # 1. 在内存中创建小型子集
    mini_df = create_mini_dataset_in_memory(
        original_file='./data/THUCNews-mini/train.txt',
        max_samples=max_samples
    )

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

    return train_loader, val_loader, tokenizer

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


def create_lora_model(num_labels, pretrained_path, lora_config=None):
    """创建LoRA微调模型（修改版）"""
    # 1. 加载配置并添加分类头信息
    config = AutoConfig.from_pretrained(
        pretrained_path,
        num_labels=num_labels
    )
    # config.update({
    #     "classifier_type": "LoRA_Enhanced",
    #     "lora_embedding_dim": 384,
    #     "lora_activation": "GELU"
    # })

    # 2. 加载基础模型
    model = AutoModelForSequenceClassification.from_pretrained(
        pretrained_path,
        config=config,
        ignore_mismatched_sizes=True
    )

    # 3. 配置LoRA参数（增强版配置）
    default_lora_config = LoraConfig(
        task_type=TaskType.SEQ_CLS,
        r=16,  # 增大秩提升能力
        lora_alpha=32, # 缩放系数
        target_modules=["query", "key", "value", "dense"],  # 覆盖更多层
        lora_dropout=0.1,
        modules_to_save=["classifier"],  # 同时微调分类头，保存时也一并保存
        bias="none"  # 不训练偏置
    )
    lora_config = lora_config or default_lora_config

    # 4. 转换为LoRA模型
    lora_model = get_peft_model(model, lora_config)

    # 5. 打印可训练参数
    # === 新增参数统计 ===
    total_params = sum(p.numel() for p in lora_model.parameters())
    trainable_params = sum(p.numel() for p in lora_model.parameters() if p.requires_grad)

    print("\n" + "=" * 40)
    print(f"总参数数量: {total_params:,}")
    print(f"可训练参数数量: {trainable_params:,} (占 {trainable_params / total_params * 100:.2f}%)")
    print("=" * 40 + "\n")

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

    # print(f"To Train Using device: {device}")

    for batch in tqdm(data_loader, desc="Training"):
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device) # 数据读入到GPU

        # 测试数据和模型是否都移动到GPU上
        # print(f"input_ids device: {input_ids.device}")
        # print(f"model device: {next(model.parameters()).device}")

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

def train_and_evaluate_lora(model, train_loader, val_loader, num_epochs=3, learning_rate=1e-4):
    """LoRA专用训练函数"""
    """LoRA专用训练函数（带显存监控）"""
    # 训练前打印显存信息
    print(f"训练前显存占用: {torch.cuda.memory_allocated() / 1024 ** 2:.2f} MB")

    # 优化器仅更新可训练参数
    optimizer = AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=learning_rate,  # LoRA通常需要更大学习率
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


    # 6. 执行训练和评估
    # --------------------------------------------------
    # 记录训练过程
    history = {
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': []
    }
    best_accuracy = 0
    for epoch in range(EPOCHS):
        print(f"\nEpoch {epoch + 1}/{EPOCHS}")
        print("-" * 10)

        # 训练
        train_loss, train_acc = train_epoch(model, train_loader, optimizer, scheduler, device)
        history['train_loss'].append(float(train_loss))
        history['train_acc'].append(float(train_acc))
        print(f"Train loss: {train_loss:.4f}, accuracy: {train_acc:.4f}")

        # 评估
        val_loss, val_acc = eval_model(model, val_loader, device)
        history['val_loss'].append(float(val_loss))
        history['val_acc'].append(float(val_acc))
        print(f"Validation loss: {val_loss:.4f}, accuracy: {val_acc:.4f}")

        # # 整理一下输出方便对比看
        # print(f" - > Train accuracy: {train_acc:.4f}, Validation accuracy: {val_acc:.4f}")

        # 检查模型是否正确训练和保存分类头
        print("模型状态字典键：")
        for key in model.state_dict().keys():
            print(key)  # 应该包含classifier和bias
        train_classifier_norm = torch.norm(model.classifier.weight).item()
        print(f"训练结束时分类头权重范数: {train_classifier_norm:.4f} , 应该和测试前加载的权重一样")

        # 保存最佳模型
        # 保存（Hugging Face推荐方式）
        # 生成config.json和pytorch_model.bin, config.json中自动包含所有配置
        model.save_pretrained(
            output_lora_model_path,
            state_dict=model.state_dict(),  # 关键：保存完整状态
            safe_serialization=True  # 添加安全序列化选项
        )
        tokenizer.save_pretrained(output_lora_model_path)  # 保存分词器
        print(f"模型已保存到 {output_lora_model_path}")

        # 学习率调整
        scheduler.step(val_acc)

        # === 新增显存监控 ===
        epoch_mem = torch.cuda.max_memory_allocated() / 1024 ** 2
        print(f"Epoch {epoch + 1} 最大显存占用: {epoch_mem:.2f} MB")
        torch.cuda.reset_peak_memory_stats()

    # === 训练完成后再合并保存 ===
    print("\n训练完成，合并最佳模型...")
    merged_model = save_merged_model(model, output_lora_model_path + "_merged")

    # 立即验证保存效果
    _, val_acc = eval_model(merged_model, val_loader, device)
    print(f"合并后模型验证准确率: {val_acc:.4f} (应与原始模型一致)")


    return best_accuracy, history


def save_merged_model(model, output_dir):
    """合并LoRA权重后保存完整模型"""
    # 合并LoRA权重到基础模型
    merged_model = model.merge_and_unload()

    # 保存完整模型
    merged_model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    print(f"合并后的完整模型已保存到 {output_dir}")

    # 验证保存效果
    print("合并后模型分类头权重范数:", torch.norm(merged_model.classifier.weight).item())

    return merged_model

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
    # plt.show()


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
# 测试新闻分类
def predict_category(model, tokenizer, text, device):
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

    # THUCNews的10个类别（示例，需替换为您的实际类别名）
    category_map = {
        0: "财经", 1: "房产", 2: "教育", 3: "科技",
        4: "军事", 5: "汽车", 6: "体育", 7: "游戏",
        8: "娱乐", 9: "政治"
    }
    # 测试情感分类
    # category_map = {0: "Negative", 1: "Neutral", 2: "Positive"}
    return category_map[pred], probs.cpu().numpy()[0]


# ==================== 4. 测试集评估 ====================
def load_test_data(file_path):
    """加载测试集数据"""
    test_df = pd.read_csv(file_path, sep='\t', header=None, names=['text', 'label'])
    print(f"\n测试集 {file_path} 的统计信息：")
    print(f"总样本数: {len(test_df)}")
    print("标签分布:")
    print(test_df['label'].value_counts())
    return test_df


# 包含最大熵样本统计和输出的测试函数
def test_model_lora(model, test_loader, device, test_df, output_dir='./test_results'):
    """
    增强版测试评估函数
    功能：
    1. 计算标准测试指标
    2. 识别预测不确定性高的样本（最大熵）
    3. 记录分类错误的样本信息到文件
    """
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    error_log_file = os.path.join(output_dir, 'error_samples.txt')
    entropy_log_file = os.path.join(output_dir, 'high_entropy_samples.txt')

    model.eval()
    total_correct = 0
    total_samples = 0
    test_loss = 0
    all_preds = []
    all_labels = []

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
                    'pred_label': preds[i].item(),
                    'probabilities': probs[i].cpu().numpy(),
                    'entropy': batch_entropy[i]
                }

                # 记录错误样本
                if preds[i] != labels[i]:
                    error_samples.append(sample_info)

                # 记录高熵样本（熵大于平均熵+标准差）
                if batch_entropy[i] > np.mean(batch_entropy) + np.std(batch_entropy):
                    high_entropy_samples.append(sample_info)

    # 计算评估指标
    avg_loss = test_loss / len(test_loader)
    accuracy = total_correct / total_samples

    # 打印基础结果
    print(f"\n测试集结果:")
    print(f"平均损失: {avg_loss:.4f}")
    print(f"准确率: {accuracy:.4f}")
    print(f"错误样本数: {len(error_samples)}/{total_samples}")
    print(f"高不确定性样本数: {len(high_entropy_samples)}/{total_samples}")

    # 分类报告
    from sklearn.metrics import classification_report
    print("\n分类报告:")
    print(classification_report(all_labels, all_preds,
                                target_names=[str(i) for i in test_df['label'].unique()]))

    # 保存错误样本到文件
    with open(error_log_file, 'w', encoding='utf-8') as f:
        f.write("=== 分类错误样本 ===\n")
        f.write(f"总数: {len(error_samples)}\n\n")
        for sample in sorted(error_samples, key=lambda x: x['entropy'], reverse=True)[:100]:  # 最多保存100个
            f.write(f"文本: {sample['text']}\n")
            f.write(f"真实标签: {sample['true_label']}\n")
            f.write(f"预测标签: {sample['pred_label']}\n")
            f.write(f"各类别概率: {np.round(sample['probabilities'], 4)}\n")
            f.write(f"熵值: {sample['entropy']:.4f}\n")
            f.write("-" * 50 + "\n")

    # 保存高熵样本到文件
    with open(entropy_log_file, 'w', encoding='utf-8') as f:
        f.write("=== 高不确定性样本 ===\n")
        f.write(f"总数: {len(high_entropy_samples)}\n\n")
        for sample in sorted(high_entropy_samples, key=lambda x: x['entropy'], reverse=True)[:100]:
            f.write(f"文本: {sample['text']}\n")
            f.write(f"真实标签: {sample['true_label']}\n")
            f.write(f"预测标签: {sample['pred_label']}\n")
            f.write(f"各类别概率: {np.round(sample['probabilities'], 4)}\n")
            f.write(f"熵值: {sample['entropy']:.4f}\n")
            f.write("-" * 50 + "\n")

    # 打印部分样本示例
    print("\n=== 分类错误样本示例 ===")
    for sample in error_samples[:3]:
        print(f"文本: {sample['text'][:50]}...")
        print(f"真实标签: {sample['true_label']} | 预测: {sample['pred_label']}")
        print(f"熵值: {sample['entropy']:.4f}\n")

    print("\n=== 高不确定性样本示例 ===")
    for sample in high_entropy_samples[:3]:
        print(f"文本: {sample['text'][:50]}...")
        print(f"预测标签: {sample['pred_label']} | 熵值: {sample['entropy']:.4f}")
        print(f"概率分布: {np.round(sample['probabilities'], 2)}\n")

    return {
        'accuracy': accuracy,
        'loss': avg_loss,
        'error_samples': error_samples,
        'high_entropy_samples': high_entropy_samples
    }



# ==================== 5. 单条样本测试示例 ====================
def predict_example(model, tokenizer, text, device):
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

    with torch.no_grad():
        outputs = model(**inputs)
        probs = torch.nn.functional.softmax(outputs.logits, dim=1)
        pred = torch.argmax(probs, dim=1).item()

    return pred, probs.cpu().numpy()[0]


def load_trained_lora_model(pretrained_path, output_lora_model_path, num_classes=10):
    """加载训练好的模型"""
    if not os.path.exists(output_lora_model_path):
        raise FileNotFoundError(f"lora模型文件 {output_lora_model_path} 不存在！")

    # 1. 加载基础配置
    config = AutoConfig.from_pretrained(
        pretrained_path,
        num_labels=num_labels # 使用实际类别数
    )
    # # 新增加了，保证和训练时候一致
    # config.update({
    #     "classifier_type": "LoRA_Enhanced",
    #     "lora_embedding_dim": 384,
    #     "lora_activation": "GELU"
    # })

    # 初始化中文模型
    base_model = AutoModelForSequenceClassification.from_pretrained(
        pretrained_path,
        config=config,
        ignore_mismatched_sizes=True,
        local_files_only=True  # 强制使用本地文件
    )

    try:
        model = PeftModel.from_pretrained(
            base_model,
            output_lora_model_path,
            is_trainable=False,
            config=LoraConfig.from_pretrained(output_lora_model_path)
        )

        # 4. 验证加载结果
        print("\n=== 权重加载验证 ===")
        print(f"分类头权重范数: {torch.norm(model.classifier.weight).item():.4f}")
        print(f"LoRA层加载: {any('lora' in n for n, _ in model.named_parameters())}")
        print(f"可训练参数: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")

        print("=== LoRA层验证 ===")
        print(f"LoRA层是否加载: {any('lora' in n for n, _ in model.named_parameters())}")
        print(f"可训练参数量: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")
        print(f"可训练参数示例: {[n for n, p in model.named_parameters() if p.requires_grad][:3]}")

    except Exception as e:
        print(f"加载失败: {str(e)}")
        # 应急方案：尝试合并权重后加载
        print("尝试加载合并后的模型...")
        model = AutoModelForSequenceClassification.from_pretrained(
            output_lora_model_path,
            config=config
        )

    loaded_classifier_norm = torch.norm(model.classifier.weight).item()
    print(f"加载后分类头权重范数: {loaded_classifier_norm:.4f}, 应该和训练结束时候的一样")

    return model.to(device)

def load_merged_model(model_path):
    """加载合并后的完整模型"""
    model = AutoModelForSequenceClassification.from_pretrained(model_path)

    print("\n=== 合并模型加载验证 ===")
    print(f"分类头权重范数: {torch.norm(model.classifier.weight).item():.4f}")
    print(f"模型结构: {type(model)}")

    return model.to(device)

    # 加载后需根据config信息重新构建分类头
    # 加载LoRA适配器
    # trained_model = PeftModel.from_pretrained(
    #     base_model,
    #     output_lora_model_path,
    #     is_trainable=False  # 测试时冻结
    # )

    # # 合并权重（可选，提升推理速度）
    # merged_model = trained_model.merge_and_unload()
    # print("\n已加载最佳模型权重")
    # print("=== LoRA层验证 ===")
    # print(f"LoRA层是否加载: {any('lora' in n for n, _ in merged_model.named_parameters())}")
    # print(f"可训练参数量: {sum(p.numel() for p in merged_model.parameters() if p.requires_grad)}")
    # print(f"可训练参数示例: {[n for n, p in merged_model.named_parameters() if p.requires_grad][:3]}")
    #
    # return merged_model.to(device)

def verify_model_loading(model, test_text):
    """验证模型是否正确加载"""
    pred, _ = predict_category(model, tokenizer, test_text, device)
    print(f"预测结果: {pred}")

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



# 主程序入口
if __name__ == "__main__":
    # === 路线1： 使用预训练模型进行全参数微调（SFT） ===
    # 1. 创建或获取数据
    # --------------------------------------------------
    # ...

    # 1. 创建模型
    # 创建LoRA模型
    print("\n创建LoRA模型...")
    # 初始化LoRA模型（替换原create_model）
    model = create_lora_model(
        num_labels=10,
        pretrained_path=model_dir+model_name,
        lora_config=LoraConfig(
            r=8,
            target_modules=["query", "value"]  # RTX 1060显存有限，减少目标模块
        )
    )
    print("完成LoRa模型创建！")

    # # 4. 加载数据，划分训练集和验证集，初始化分词器
    # # --------------------------------------------------
    train_loader, val_loader, tokenizer = prepare_datasets()
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


    # 5. 设置优化器和学习率调度器
    # --------------------------------------------------
    # 6. 执行训练和评估
    # -------------------------------------------------
    print("\n开始训练...")
    best_val_acc, history = train_and_evaluate_lora(
        model,
        train_loader,
        val_loader,
        num_epochs=num_epochs,
        learning_rate=learning_rate  # 您当前使用的学习率
    )




    # 训练后立即验证
    print("\n=== 训练后即时验证 ===")
    val_loss, val_acc = eval_model(model, val_loader, device)
    print(f"验证集结果 - Loss: {val_loss:.4f}, Acc: {val_acc:.4f}")

    # 保存前打印模型信息
    print("\n=== 模型状态 ===")
    print(f"分类头权重: {model.classifier.weight.data[:1]}")
    print(f"LoRA参数: {[(n, p.shape) for n, p in model.named_parameters() if 'lora' in n]}")

    # 在主程序训练后调用，训练结果可视化输出（可选）
    plot_training_history(history)

    end_time1 = time.time()
    elapsed_time1 = end_time1 - start_time
    print(f"训练耗时: {elapsed_time1} secs!")



    # --------------- 测试脚本（假设模型已训练并保存为'best_text_model.bin'） ----------------

    # ==================== 1. 加载测试集 ====================
    print("\n加载测试集...")
    # 加载测试集（假设路径为'./data/THUCNews-mini/test.txt'）
    test_df = load_THUCNews_test(test_data_path)

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
    # test_model = load_trained_model(model_dir+model_name, num_labels)
    # test_model = load_trained_lora_model(model_dir+model_name, output_lora_model_path, num_labels)
    test_model = load_merged_model(output_lora_model_path + "_merged")

    # 测试是否正确加载了模型
    # 使用示例
    test_text = "苹果发布新款iPhone"
    verify_model_loading(test_model, test_text)

    # 执行测试
    # test_model(model, test_loader, device)
    # === 输出最大熵测试样本
    test_model_lora(test_model, test_loader, device, test_df, output_dir=ouput_dir)


    # === 测试几条样本
    test_samples = [
        "国足再次失利，球迷表示失望",  # 示例1
        "央行宣布降准0.5个百分点",  # 示例2
        "新款iPhone发布，起售价5999元"  # 示例3
    ]
    print("\n单条样本预测示例:")
    for sample in test_samples:
        pred_label, pred_probs = predict_example(test_model, tokenizer, sample, device)
        print(f"文本: {sample}")
        print(f"预测标签: {pred_label}")
        print(f"各类别概率: {pred_probs.round(4)}")
        print("-" * 50)


    print('\n---')
    total_time = time.time() - start_time
    print(f"Total time: {total_time} secs!")



