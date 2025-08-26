'''
    以下是一个完整的代码示例，用于在个人笔记本的GeForce RTX 1060上通过主动学习微调轻量级文本模型（DistilBERT）进行多分类任务。
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


    by 李明华，2025-07-29.
'''


import torch
from torch.utils.data import Dataset, DataLoader
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
from transformers import AdamW, get_linear_schedule_with_warmup
import numpy as np
from sklearn.metrics import accuracy_score
from tqdm import tqdm
import random
import sys
import time

# 检查GPU可用性
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# 记录开始时间
start_time = time.time()

# 本地模型文件加载测试（在 Docker 容器中使用）
MODEL_PATH = "./models/distilbert-base-uncased" # 在代码中指定本地路径
tokenizer = DistilBertTokenizer.from_pretrained(MODEL_PATH)
model = DistilBertForSequenceClassification.from_pretrained(
    MODEL_PATH,
    num_labels=3
)

print('Successfully loaded!')

sys.exit()



# 1. 创建小型训练数据集 - 电影评论情感分析（3分类）
class MovieReviewDataset(Dataset):
    """
    自定义数据集类，用于存储电影评论和情感标签
    标签: 0=负面, 1=中性, 2=正面
    """

    def __init__(self, reviews, labels, tokenizer, max_length=128):
        self.reviews = reviews
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.reviews)

    def __getitem__(self, idx):
        review = str(self.reviews[idx])
        label = self.labels[idx]

        # 使用tokenizer对文本进行编码
        encoding = self.tokenizer.encode_plus(
            review,
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


# 少量训练数据（40个样本）
reviews = [
    "This movie was absolutely fantastic! The acting was superb.",
    "I found the plot to be confusing and poorly developed.",
    "The cinematography was stunning, but the story was weak.",
    "A mediocre film with nothing special to offer.",
    "One of the best movies I've seen this year!",
    "The characters were poorly developed and uninteresting.",
    "Great performances from the entire cast.",
    "The ending was disappointing and predictable.",
    "A masterpiece of modern cinema.",
    "Not worth the price of admission.",
    "The director did an excellent job with this material.",
    "I was bored throughout the entire film.",
    "The special effects were impressive but overused.",
    "A heartwarming story that everyone should see.",
    "The script was full of clichés and tired tropes.",
    "An unforgettable cinematic experience.",
    "I couldn't connect with any of the characters.",
    "The soundtrack perfectly complemented the visuals.",
    "A complete waste of time and money.",
    "This film exceeded all my expectations.",
    "The pacing was too slow for my taste.",
    "Brilliant dialogue and character development.",
    "I walked out halfway through, it was that bad.",
    "A thought-provoking exploration of complex themes.",
    "The humor fell flat and felt forced.",
    "Visually spectacular but emotionally hollow.",
    "A solid film that delivers what it promises.",
    "I've seen better student films than this.",
    "The lead actor gave a career-best performance.",
    "The plot twists were predictable from miles away.",
    "A true gem that deserves more recognition.",
    "I regret spending two hours of my life on this.",
    "The chemistry between the leads was electric.",
    "Poorly edited with jarring scene transitions.",
    "A refreshing take on a familiar genre.",
    "The movie tried too hard to be profound.",
    "Perfect for a relaxing evening at home.",
    "I wouldn't recommend this to my worst enemy.",
    "An emotional rollercoaster from start to finish.",
    "Technically competent but utterly forgettable."
]

# 标签: 0=负面, 1=中性, 2=正面
labels = [
    2, 0, 1, 1, 2,
    0, 2, 0, 2, 0,
    2, 0, 1, 2, 0,
    2, 0, 2, 0, 2,
    0, 2, 0, 2, 0,
    1, 1, 0, 2, 0,
    2, 0, 2, 0, 2,
    0, 1, 0, 2, 1
]

# 2. 初始化分词器和模型
MODEL_NAME = 'distilbert-base-uncased'
# # 联网下载模型
# tokenizer = DistilBertTokenizer.from_pretrained(MODEL_NAME)
# model = DistilBertForSequenceClassification.from_pretrained(
#     MODEL_NAME,
#     num_labels=3  # 3分类任务
# ).to(device)
# 本地模型文件加载（在 Docker 容器中使用）
MODEL_PATH = "./models/distilbert-base-uncased" # 在代码中指定本地路径
tokenizer = DistilBertTokenizer.from_pretrained(MODEL_PATH)
model = DistilBertForSequenceClassification.from_pretrained(
    MODEL_PATH,
    num_labels=3
)

# 3. 准备数据加载器
dataset = MovieReviewDataset(reviews, labels, tokenizer)
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=8)

# 4. 设置优化器和学习率调度器
EPOCHS = 3  # 少量epochs用于演示
optimizer = AdamW(model.parameters(), lr=2e-5, correct_bias=False)
total_steps = len(train_loader) * EPOCHS
scheduler = get_linear_schedule_with_warmup(
    optimizer,
    num_warmup_steps=0,
    num_training_steps=total_steps
)


# 5. 训练循环
def train_epoch(model, data_loader, optimizer, scheduler, device):
    model.train()
    losses = []
    correct_predictions = 0

    for batch in tqdm(data_loader, desc="Training"):
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


# 7. 执行训练和评估
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

    # 保存最佳模型
    if val_acc > best_accuracy:
        torch.save(model.state_dict(), 'best_model.bin')
        best_accuracy = val_acc


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
    unlabeled_reviews = reviews.copy()
    unlabeled_labels = labels.copy()

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
    train_dataset = MovieReviewDataset(train_reviews, train_labels, tokenizer)
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
        train_dataset = MovieReviewDataset(train_reviews, train_labels, tokenizer)
        train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)

        for epoch in range(EPOCHS):
            train_loss, train_acc = train_epoch(model, train_loader, optimizer, scheduler, device)
            print(f"Epoch {epoch + 1}: Train loss: {train_loss:.4f}, accuracy: {train_acc:.4f}")

        print(f"Training set size: {len(train_reviews)} samples")
        print(f"Unlabeled pool size: {len(unlabeled_reviews)} samples")


# 9. 执行主动学习循环
print("\nStarting Active Learning Process")
active_learning_cycle(model, tokenizer, device)

# 10. 保存最终模型
torch.save(model.state_dict(), 'final_model.bin')
print("\nTraining complete! Model saved as 'final_model.bin'")


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


# 测试新评论
test_reviews = [
    "This movie was a delightful surprise!",
    "I expected more from such a talented cast.",
    "It was okay, but nothing special."
]

print("\nModel Predictions:")
for review in test_reviews:
    sentiment, probs = predict_sentiment(model, tokenizer, review, device)
    print(f"Review: {review}")
    print(
        f"Sentiment: {sentiment} | Probabilities: Negative={probs[0]:.2f}, Neutral={probs[1]:.2f}, Positive={probs[2]:.2f}")
    print("-" * 50)


end_time = time.time()
elapsed_time = end_time - start_time
print(f"Elapsed time: {elapsed_time} secs!")

