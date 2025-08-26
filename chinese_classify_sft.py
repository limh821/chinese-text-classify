'''
    以下是一个完整的代码示例，用于在个人笔记本的GeForce RTX 1060上通过SFT微调/主动学习微调轻量级中文文本模型（chinese-roberta-wwm-ext）进行中文文本的多分类任务。
    包含完整的代码、注释和少量训练集样本。

    这个代码设计为在NVIDIA GeForce RTX 1060（显存6GB）上高效运行，主动学习过程可以帮助你理解如何逐步改进模型性能，同时保持计算资源在可管理范围内。

    by 李明华，2025-08-16.
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
num_epochs = 15
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
    """中文电影评论情感分析数据集"""

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
    assert train_df['label'].max() == (num_labels-1), "警告：训练集中的标签数目与指定的num_labels数目不同！标签需转换为0-(num_labels)的连续整数"

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

def create_model(num_labels, pretrained_path=None, freeze_level=0):
    """ 创建并配置模型 """
    # MODEL_NAME = 'chinese-roberta-wwm-ext'
    # # 联网下载模型
    # tokenizer = DistilBertTokenizer.from_pretrained(MODEL_NAME)
    # model = DistilBertForSequenceClassification.from_pretrained(
    #     MODEL_NAME,
    #     num_labels=3  # 3分类任务
    # ).to(device)
    # 本地模型文件加载（可在 Docker 容器中使用）
    # MODEL_PATH = pretrained_path # 在代码中指定本地路径
    # # 初始化英文分词器和模型
    # tokenizer = DistilBertTokenizer.from_pretrained(MODEL_PATH)
    # model = DistilBertForSequenceClassification.from_pretrained(
    #     MODEL_PATH,
    #     num_labels=3
    # )
    # # 1. 快速初始化中文模型
    # model = AutoModelForSequenceClassification.from_pretrained(
    #     pretrained_path,
    #     local_files_only=True,  # 强制使用本地文件
    #     num_labels=num_labels  # 分类数量
    # )

    # 确保使用本地路径
    if not pretrained_path:
        raise ValueError("必须提供本地模型路径")

    # 1. 从本地加载配置
    config = AutoConfig.from_pretrained(
        pretrained_path,  # 关键修改：使用本地路径而非模型名称
        local_files_only=True,
        num_labels=num_labels
    )

    # 添加自定义分类头配置
    hidden_dim = 384  # 定义中间层维度
    config.update({
        "classifier_type": "EnhancedClassifier",
        "classifier_hidden_dim": hidden_dim,
        "classifier_activation": "GELU"
    })

    # 2. 初始化模型（含新配置）
    model = AutoModelForSequenceClassification.from_pretrained(
        pretrained_path,
        config=config,
        local_files_only=True,
        ignore_mismatched_sizes=True
    )

    # 检查预训练参数是否加载正确
    # todo ...

    # 修改前分类头
    print("修改前分类头：")
    print(model.classifier)  # 查看原始层的输入/输出维度

    # 根据类别数量和训练集大小，调整分类头结构和参数量
    # 方案一： 分类头增加中间层
    hidden_size = model.config.hidden_size
    model.classifier = EnhancedClassifier(
        hidden_size=hidden_size,
        hidden_dim=hidden_dim,
        num_labels=num_labels
    )

    # 修改前分类头
    print("修改后分类头：")
    print(model.classifier)  # 查看原始层的输入/输出维度

    # 3. 冻结参数
    # --------------------------------------------------
    print("冻结基础模型参数...")
    total_params = 0
    trainable_params = 0
    # 冻结roberta模块，只训练分类头
    for name, param in model.named_parameters():
        total_params += param.numel()
        if 'classifier' not in name:  # 只保留分类层可训练
            param.requires_grad = False
        else:
            trainable_params += param.numel()
            param.requires_grad = True

    print(f"总参数: {total_params:,}")
    print(f"可训练参数: {trainable_params:,} ({(trainable_params / total_params) * 100:.2f}%)")
    # 模型参数移动到GPU设备上
    model = model.to(device)

    return model

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

def train_and_evaluate_model(model, train_loader, val_loader, num_epochs=3, learning_rate=0.001):
    # 5. 设置优化器和学习率调度器
    # --------------------------------------------------
    # # === 少量epochs用于演示 ===
    # EPOCHS = 3
    # optimizer = AdamW(model.parameters(), lr=2e-5, correct_bias=False)
    # # === 修改训练参数适应中文模型 ===
    # EPOCHS = 4  # 增加epoch数
    # optimizer = AdamW(model.parameters(), lr=3e-5, weight_decay=0.01)  # 调整学习率
    # === 仅做分类头的训练（仅优化可训练参数） ===
    EPOCHS = num_epochs  # 增加epoch数
    optimizer = AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),  # 只优化需要梯度的参数
        lr=learning_rate,  # 分类头使用更大学习率
        weight_decay=weight_decay
    )
    total_steps = len(train_loader) * EPOCHS
    print('训练步数: ', total_steps)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=total_steps
    )
    # # 其他学习率调度器（可选）
    # scheduler = optim.lr_scheduler.ReduceLROnPlateau(
    #     optimizer, mode='max', factor=0.1, patience=3, verbose=True
    # )

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

        # 保存最佳模型
        # 保存（Hugging Face推荐方式）
        model.save_pretrained(output_model_path)  # 生成config.json和pytorch_model.bin, config.json中自动包含所有配置
        tokenizer.save_pretrained(output_model_path)  # 可选，保存分词器

        # 学习率调整
        scheduler.step(val_acc)

    return best_accuracy, history

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
    plt.savefig(train_history_filepath)
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



 # 8. 主动学习循环（简化版）
def active_learning_cycle(model, tokenizer, device, initial_samples=5, iterations=3, samples_per_iter=5):
    """
    主动学习循环：
    1. 使用少量样本初始化模型
    2. 在未标注数据上预测并选择最不确定的样本
    3. 人工标注这些样本
    4. 使用新样本更新模型
    """
    # 模拟未标注池（使用原始数据作为未标注数据）
    df = pd.read_csv('./data/movie_reviews.csv')
    reviews = df['text']
    labels = df['label']

    unlabeled_reviews = reviews.copy().tolist() # 转换成list，避免后面的索引错误（pandas的series索引方式和List不一样）
    unlabeled_labels = labels.copy().tolist()

    # 初始训练集（随机选择少量样本）
    initial_indices = random.sample(range(len(unlabeled_reviews)), initial_samples)
    train_reviews = [unlabeled_reviews[i] for i in initial_indices]
    train_labels = [unlabeled_labels[i] for i in initial_indices]

    # 从未标注池中移除初始样本
    for i in sorted(initial_indices, reverse=True):
        del unlabeled_reviews[i]
        del unlabeled_labels[i]

    # 训练初始模型
    print("\n=== Training Initial Model ===")
    train_dataset = ChineseMovieReviewDataset(train_reviews, train_labels, tokenizer)
    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)

    for epoch in range(EPOCHS):
        train_loss, train_acc = train_epoch(model, train_loader, optimizer, scheduler, device)
        print(f"Epoch {epoch + 1}: Train loss: {train_loss:.4f}, accuracy: {train_acc:.4f}")

    # 主动学习迭代
    for iteration in range(iterations):
        print(f"\n=== Active Learning Iteration {iteration + 1}/{iterations} ===")

        # 在未标注数据上预测并计算不确定性
        uncertainties = []
        model.eval()

        with torch.no_grad():
            for review in unlabeled_reviews:
                inputs = tokenizer(
                    review,
                    add_special_tokens=True,
                    max_length=128,
                    padding='max_length',
                    truncation=True,
                    return_tensors='pt'
                ).to(device)

                outputs = model(**inputs)
                probs = torch.nn.functional.softmax(outputs.logits, dim=1)

                # 使用熵作为不确定性度量
                entropy = -torch.sum(probs * torch.log(probs + 1e-10), dim=1)
                uncertainties.append(entropy.item())

        # 选择最不确定的样本
        most_uncertain_indices = np.argsort(uncertainties)[-samples_per_iter:][::-1]

        # 打印最不确定的样本
        print("\nMost uncertain samples:")
        for idx in most_uncertain_indices:
            print(f"- {unlabeled_reviews[idx]} (Uncertainty: {uncertainties[idx]:.4f})")

        # 将这些样本添加到训练集（模拟人工标注）
        new_reviews = [unlabeled_reviews[i] for i in most_uncertain_indices]
        new_labels = [unlabeled_labels[i] for i in most_uncertain_indices]

        train_reviews.extend(new_reviews)
        train_labels.extend(new_labels)

        # 从未标注池中移除已标注样本
        for i in sorted(most_uncertain_indices, reverse=True):
            del unlabeled_reviews[i]
            del unlabeled_labels[i]

        # 用新数据更新模型
        print(f"\nUpdating model with {len(new_reviews)} new samples...")
        train_dataset = ChineseMovieReviewDataset(train_reviews, train_labels, tokenizer)
        train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)

        for epoch in range(EPOCHS):
            train_loss, train_acc = train_epoch(model, train_loader, optimizer, scheduler, device)
            print(f"Epoch {epoch + 1}: Train loss: {train_loss:.4f}, accuracy: {train_acc:.4f}")

        print(f"Training set size: {len(train_reviews)} samples")
        print(f"Unlabeled pool size: {len(unlabeled_reviews)} samples")


def active_learning_cycle_v2(model, tokenizer, device, initial_samples=100, iterations=3, samples_per_iter=50):
    """
    主动学习循环（适配THUCNews 10分类任务）
    参数:
        initial_samples: 初始训练样本数
        iterations: 主动学习迭代次数
        samples_per_iter: 每次迭代新增样本数
    """
    # 1. 加载原始数据（模拟未标注池）
    original_file = './data/THUCNews-mini/train.txt'
    df = pd.read_csv(original_file, sep='\t', header=None, names=['text', 'label'])

    # 2. 初始化训练集和未标注池
    unlabeled_texts = df['text'].tolist()
    unlabeled_labels = df['label'].tolist()

    # 分层抽样初始训练集（保持类别分布）
    train_texts, train_labels = [], []
    label_counts = df['label'].value_counts()
    samples_per_label = (label_counts / label_counts.sum() * initial_samples).round().astype(int)

    for label in label_counts.index:
        indices = [i for i, lbl in enumerate(unlabeled_labels) if lbl == label]
        selected = np.random.choice(indices, samples_per_label[label], replace=False)
        train_texts.extend([unlabeled_texts[i] for i in selected])
        train_labels.extend([unlabeled_labels[i] for i in selected])

        # 从未标注池移除
        for i in sorted(selected, reverse=True):
            del unlabeled_texts[i]
            del unlabeled_labels[i]

    # 3. 主循环
    for iteration in range(iterations):
        print(f"\n=== 主动学习迭代 {iteration + 1}/{iterations} ===")

        # 训练模型
        train_dataset = ChineseMovieReviewDataset(train_texts, train_labels, tokenizer)
        train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)

        optimizer = AdamW(model.parameters(), lr=5e-5)
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=0,
            num_training_steps=len(train_loader) * 1  # 每个迭代只训练1个epoch
        )

        train_loss, train_acc = train_epoch(model, train_loader, optimizer, scheduler, device)
        print(f"训练准确率: {train_acc:.4f}")

        # 计算未标注样本的不确定性（基于预测熵）
        uncertainties = []
        model.eval()

        with torch.no_grad():
            for text in tqdm(unlabeled_texts, desc="计算不确定性"):
                inputs = tokenizer(
                    text,
                    add_special_tokens=True,
                    max_length=256,
                    padding='max_length',
                    truncation=True,
                    return_tensors='pt'
                ).to(device)

                outputs = model(**inputs)
                probs = torch.nn.functional.softmax(outputs.logits, dim=1)
                entropy = -torch.sum(probs * torch.log(probs + 1e-10), dim=1)
                uncertainties.append(entropy.item())

        # 选择最不确定的样本
        selected_indices = np.argsort(uncertainties)[-samples_per_iter:][::-1]

        # 打印最不确定的样本示例
        print("\n最不确定的5个样本:")
        for idx in selected_indices[:5]:
            print(f"文本: {unlabeled_texts[idx][:50]}... | 不确定性: {uncertainties[idx]:.4f}")

        # 添加到训练集
        train_texts.extend([unlabeled_texts[i] for i in selected_indices])
        train_labels.extend([unlabeled_labels[i] for i in selected_indices])

        # 从未标注池移除
        for i in sorted(selected_indices, reverse=True):
            del unlabeled_texts[i]
            del unlabeled_labels[i]

        print(f"\n当前训练集大小: {len(train_texts)}")
        print(f"剩余未标注样本: {len(unlabeled_texts)}")

        # 验证集评估
        val_dataset = ChineseMovieReviewDataset(
            train_texts[-100:],  # 用最新样本作为验证
            train_labels[-100:],
            tokenizer
        )
        val_loader = DataLoader(val_dataset, batch_size=16)
        val_loss, val_acc = eval_model(model, val_loader, device)
        print(f"验证准确率: {val_acc:.4f}")


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

def test_model(model, test_loader, device):
    """在测试集上评估模型"""
    model.eval()
    total_correct = 0
    total_samples = 0
    test_loss = 0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Testing"):
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

            _, preds = torch.max(logits, dim=1)
            total_correct += (preds == labels).sum().item()
            total_samples += labels.size(0)
            test_loss += loss.item()

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    avg_loss = test_loss / len(test_loader)
    accuracy = total_correct / total_samples

    print(f"\n测试集结果:")
    print(f"平均损失: {avg_loss:.4f}")
    print(f"准确率: {accuracy:.4f}")

    # 添加分类报告（需要sklearn）
    from sklearn.metrics import classification_report
    print("\n分类报告:")
    print(classification_report(all_labels, all_preds, target_names=[str(i) for i in test_df['label'].unique()]))

# 包含最大熵样本统计和输出的测试函数
def test_model_v2(model, test_loader, device, test_df, output_dir='./test_results'):
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


def load_trained_model(model_path, num_classes=10):
    """加载训练好的模型"""
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"模型文件 {model_path} 不存在！")

    # 初始化中文模型
    loaded_model = AutoModelForSequenceClassification.from_pretrained(
        model_path,
        local_files_only=True,  # 强制使用本地文件
        num_labels=len(test_df['label'].unique()) # 使用实际类别数
    )

    # 加载后需根据config信息重新构建分类头
    if hasattr(loaded_model.config, 'classifier_type'):
        loaded_model.classifier = EnhancedClassifier()

    # # 如果只有.pth文件，需先初始化模型再加载权重：
    # model.load_state_dict(torch.load(model_path))
    # model = model.to(device)

    print("\n已加载最佳模型权重")
    return model.to(device)








# 主程序入口
if __name__ == "__main__":
    # === 路线1： 使用预训练模型进行全参数微调（SFT） ===
    # 1. 创建或获取数据
    # --------------------------------------------------
    # ...

    # 1. 创建模型
    print("\n创建模型...")
    model = create_model(num_labels, pretrained_path=model_dir+model_name, freeze_level=0)
    print("完成模型创建！")

    # 4. 加载数据，划分训练集和验证集，初始化分词器
    # --------------------------------------------------
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
    best_val_acc, history = train_and_evaluate_model(
        model,
        train_loader,
        val_loader,
        num_epochs=num_epochs,
        learning_rate=learning_rate  # 您当前使用的学习率
    )

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
    test_model = load_trained_model(model_dir+model_name, num_labels)


    # 执行测试
    # test_model(model, test_loader, device)
    # === 输出最大熵测试样本
    test_model_v2(test_model, test_loader, device, test_df, output_dir=ouput_dir)


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



