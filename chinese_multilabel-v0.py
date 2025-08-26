'''
    以下是一个完整的代码示例，用于在个人笔记本的GeForce RTX 1060上通过主动学习微调轻量级中文文本模型（chinese-roberta-wwm-ext）进行中文文本的多分类任务。
    包含完整的代码、注释和少量训练集样本。

    这个代码设计为在NVIDIA GeForce RTX 1060（显存6GB）上高效运行，主动学习过程可以帮助你理解如何逐步改进模型性能，同时保持计算资源在可管理范围内。

    预期结果：
    Epoch 1/3
    ----------
    Training: 100%|██████████| 4/4 [00:01<00:00,  3.23it/s]
    Train loss: 1.0805, accuracy: 0.4375
    Validation: 100%|██████████| 1/1 [00:00<00:00,  6.02it/s]
    Validation loss: 0.9982, accuracy: 0.6250

    Starting Active Learning Process

    === Active Learning Iteration 1/3 ===
    Most uncertain samples:
    - The cinematography was stunning, but the story was weak. (Uncertainty: 1.0569)
    - A solid film that delivers what it promises. (Uncertainty: 1.0392)
    - ...

    Model Predictions:
    Review: This movie was a delightful surprise!
    Sentiment: Positive | Probabilities: Negative=0.12, Neutral=0.23, Positive=0.65


    by 李明华，2025-07-27.
'''


import torch
from torch.utils.data import Dataset, DataLoader
# 英文分词器
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
# 中文分词器
from transformers import AutoTokenizer, AutoModelForSequenceClassification
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

# 检查GPU可用性
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# 设置为离线模式
os.environ['TRANSFORMERS_OFFLINE'] = '1'

# 训练超参数设置
train_path = './data/movie_reviews.csv'
batch_size = 32
num_epochs = 5
learning_rate = 1e-3
weight_decay = 0.01







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

def load_and_split_THUCNews_data(file_path, test_size=0.2, random_state=42):
    """加载CSV数据并分割为训练集和验证集"""
    # 读取Tab分隔的.txt文件
    df = pd.read_csv(file_path, sep='\t', header=None, names=['text', 'label'])
    print(f"训练集 {file_path} 的表头： ", df.head())

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




# 数据处理流程
def prepare_datasets(model_name='./models/chinese-roberta-wwm-ext'):
    """完整数据准备流程"""
    # 1. 加载并分割数据
    train_df, val_df = load_and_split_data(train_path)
    # # 针对THUCNew-mini数据集的加载
    # train_df, val_df = load_and_split_THUCNews_data('./data/THUCNews-mini/train.txt')


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
        batch_size=8,
        shuffle=True,
        num_workers=2
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=8,
        shuffle=False,
        num_workers=2
    )

    return train_loader, val_loader, tokenizer

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

# 6. 评估函数
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


# 11. 测试模型
def predict_sentiment(model, tokenizer, text, device):
    """预测单个文本的情感"""
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

    sentiment_map = {0: "Negative", 1: "Neutral", 2: "Positive"}
    return sentiment_map[pred], probs.cpu().numpy()[0]




# 主程序入口
if __name__ == "__main__":
    # === 路线1： 使用预训练模型进行全参数微调（SFT） ===
    # 1. 创建或获取数据
    # --------------------------------------------------
    # ...

    # 2. 初始化模型
    # --------------------------------------------------
    MODEL_NAME = 'chinese-roberta-wwm-ext'
    # # 联网下载模型
    # tokenizer = DistilBertTokenizer.from_pretrained(MODEL_NAME)
    # model = DistilBertForSequenceClassification.from_pretrained(
    #     MODEL_NAME,
    #     num_labels=3  # 3分类任务
    # ).to(device)
    # 本地模型文件加载（可在 Docker 容器中使用）
    MODEL_PATH = "./models/"+MODEL_NAME # 在代码中指定本地路径
    # # 初始化英文分词器和模型
    # tokenizer = DistilBertTokenizer.from_pretrained(MODEL_PATH)
    # model = DistilBertForSequenceClassification.from_pretrained(
    #     MODEL_PATH,
    #     num_labels=3
    # )
    # 初始化中文模型
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_PATH,
        local_files_only=True,  # 强制使用本地文件
        num_labels=3  # 分类数量
    )

    # 3. 冻结参数
    # --------------------------------------------------
    print("冻结基础模型参数...")
    total_params = 0
    trainable_params = 0
    # 冻结roberta模块，只训练分类头
    for name, param in model.named_parameters():
        if 'classifier' not in name:  # 只保留分类层可训练
            param.requires_grad = False
            total_params += param.numel()
        else:
            trainable_params += param.numel()
            param.requires_grad = True
    # === 其他冻结策略（可选） ===
    # # 完全冻结基础模型
    # for param in model.roberta.parameters():
    #     param.requires_grad = False
    # # 解冻最后2层Transformer
    # for i in [10, 11]:  # 最后两层
    #     for param in model.roberta.encoder.layer[i].parameters():
    #         param.requires_grad = True
    # # 解冻池化层
    # for param in model.roberta.pooler.parameters():
    #     param.requires_grad = True
    # # 解冻分类头
    # for param in model.classifier.parameters():
    #     param.requires_grad = True
    print(f"总参数: {total_params + trainable_params:,}")
    print(f"可训练参数: {trainable_params:,} ({(trainable_params / (total_params + trainable_params)) * 100:.2f}%)")
    # 模型参数移动到GPU设备上
    model = model.to(device)

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
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=0,
        num_training_steps=total_steps
    )
    # # === 分组设置学习率（可选） ===
    # optimizer_param_groups = [
    #     # 最后两层Transformer
    #     {'params': model.roberta.encoder.layer[10].parameters(), 'lr': 1e-5},
    #     {'params': model.roberta.encoder.layer[11].parameters(), 'lr': 1e-5},
    #     # 池化层
    #     {'params': model.roberta.pooler.parameters(), 'lr': 5e-5},
    #     # 分类头
    #     {'params': model.classifier.parameters(), 'lr': 1e-4}
    # ]
    # optimizer = AdamW(optimizer_param_groups)
    # # === 基于不同epoch循环的、更复杂的参数解冻调度（可选） ===
    # def unfreeze_layers(model, current_epoch, total_epochs):
    #     """随着训练进度逐步解冻层"""
    #     # 计算解冻比例
    #     unfreeze_ratio = current_epoch / total_epochs
    #     # 计算解冻层数
    #     total_layers = len(model.roberta.encoder.layer)
    #     unfreeze_up_to = int(total_layers * unfreeze_ratio)
    #     # 冻结所有层
    #     for param in model.roberta.parameters():
    #         param.requires_grad = False
    #     # 逐步解冻
    #     for i in range(total_layers - unfreeze_up_to, total_layers):
    #         for param in model.roberta.encoder.layer[i].parameters():
    #             param.requires_grad = True
    #     # 始终解冻分类头
    #     for param in model.classifier.parameters():
    #         param.requires_grad = True
    # # === 不同epoch循环中，先解冻后冻结（可选） ===
    # for epoch in range(EPOCHS):
    #     # 解冻策略
    #     unfreeze_layers(model, epoch, EPOCHS)
    #     # 重新创建优化器
    #     optimizer = AdamW(
    #         filter(lambda p: p.requires_grad, model.parameters()),
    #         lr=1e-4
    #     )
    #     # 训练步骤...

    # 6. 执行训练和评估
    # --------------------------------------------------
    best_accuracy = 0
    for epoch in range(EPOCHS):
        print(f"\nEpoch {epoch + 1}/{EPOCHS}")
        print("-" * 10)

        # 训练
        train_loss, train_acc = train_epoch(model, train_loader, optimizer, scheduler, device)
        print(f"Train loss: {train_loss:.4f}, accuracy: {train_acc:.4f}")

        # 评估
        val_loss, val_acc = eval_model(model, val_loader, device)
        print(f"Validation loss: {val_loss:.4f}, accuracy: {val_acc:.4f}")

        # 整理一下输出方便对比看
        print(f" - > Train accuracy: {train_acc:.4f}, Validation accuracy: {val_acc:.4f}")

        # 保存最佳模型
        if val_acc > best_accuracy:
            torch.save(model.state_dict(), 'best_model.bin')
            best_accuracy = val_acc

    end_time1 = time.time()
    elapsed_time1 = end_time1 - start_time
    print(f"训练耗时: {elapsed_time1} secs!")


    # # === 路线2： 结合主动学习和人工增量标注的SFT（可选） ===
    # print("\nStarting Active Learning Process")
    # active_learning_cycle(model, tokenizer, device)
    # # 保存最终模型
    # torch.save(model.state_dict(), 'final_model.bin')
    # print("\nTraining complete! Model saved as 'final_model.bin'")

    # === 测试新评论 ===
    test_reviews = [
        "全程无尿点!超出预期！",
        "值得去电影院看一下。",
        "超级烂片！还是多引进一些美国大片。"
    ]
    # THUCNews_test = [
    #     "词汇阅读是关键 08年考研暑期英语复习全指南",
    #     "陈法蓉豪门梦碎 男友负债累累玩失踪",
    #     "抛储预期打压农产品价格",
    #     "街机新作《拳皇13》发表 详情3月公布"
    # ]
    # test_reviews = THUCNews_test
    print("\nModel Predictions:")
    for review in test_reviews:
        sentiment, probs = predict_sentiment(model, tokenizer, review, device)
        print(f"Review: {review}")
        print(
            f"Sentiment: {sentiment} | Probabilities: Negative={probs[0]:.2f}, Neutral={probs[1]:.2f}, Positive={probs[2]:.2f}")
        print("-" * 50)


    print('\n---')
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Totally Elapsed time: {elapsed_time} secs!")

