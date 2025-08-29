'''
    实现文本训练集的快速分层抽样，支持百万级文本的秒级分层抽样。

    创建 by 李明华，2025-08-29.

'''
import os.path

import pandas as pd
import numpy as np
from collections import defaultdict
import time

# 输入数据
input_data_path = '../data/THUCNews-mini/train.txt'
# 抽样数目
max_samples = 5000
# 输出目录
output_dir = '../data/preprocess/'
output_data_path = os.path.join(output_dir, 'mini_train_set.csv')


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
    # 读取原始数据
    print("正在读取数据...")
    start_time = time.time()
    df = pd.read_csv(original_file, sep='\t', header=None, names=['text', 'label'])
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


# 备用方案：如果数据量特别大，可以使用逐块读取的方式
def create_mini_dataset_chunked(original_file, max_samples=1000, random_state=42, chunksize=10000):
    """
    针对超大文件的逐块读取版本，避免内存不足
    """
    if random_state is not None:
        np.random.seed(random_state)

    print("使用逐块读取方式处理超大文件...")

    # 第一遍：统计标签分布
    label_counts = {}
    total_count = 0

    for chunk in pd.read_csv(original_file, sep='\t', header=None, names=['text', 'label'], chunksize=chunksize):
        chunk_counts = chunk['label'].value_counts().to_dict()
        for label, count in chunk_counts.items():
            label_counts[label] = label_counts.get(label, 0) + count
        total_count += len(chunk)

    print(f"文件统计完成：总共 {total_count} 条记录，{len(label_counts)} 个类别")

    # 计算每个类别的抽样数量
    sample_sizes = {}
    for label, count in label_counts.items():
        sample_sizes[label] = max(1, int(max_samples * count / total_count))

    # 调整总数
    total_selected = sum(sample_sizes.values())
    if total_selected != max_samples:
        diff = max_samples - total_selected
        sorted_labels = sorted(label_counts.items(), key=lambda x: x[1], reverse=True)
        for label, count in sorted_labels:
            if diff == 0:
                break
            available = count - sample_sizes.get(label, 0)
            if available > 0:
                to_add = min(diff, available)
                sample_sizes[label] = sample_sizes.get(label, 0) + to_add
                diff -= to_add

    # 第二遍：实际抽样
    sampled_data = {label: [] for label in sample_sizes.keys()}

    for chunk in pd.read_csv(original_file, sep='\t', header=None, names=['text', 'label'], chunksize=chunksize):
        for label, size in sample_sizes.items():
            if size > 0:
                label_data = chunk[chunk['label'] == label]
                if len(label_data) > 0:
                    if len(sampled_data[label]) < size:
                        needed = size - len(sampled_data[label])
                        sample = label_data.sample(min(needed, len(label_data)), random_state=random_state)
                        sampled_data[label].extend(sample.to_dict('records'))

    # 合并结果
    all_samples = []
    for samples in sampled_data.values():
        all_samples.extend(samples)

    mini_df = pd.DataFrame(all_samples)
    if len(mini_df) > max_samples:
        mini_df = mini_df.sample(max_samples, random_state=random_state)

    print(f"创建小型训练子集完成: 共 {len(mini_df)} 条样本")
    return mini_df


# 使用示例
if __name__ == "__main__":
    start_time_0 = time.time()

    # 测试高效版本
    mini_train_df = create_mini_dataset_fast(input_data_path, max_samples=max_samples, random_state=42)

    # 打印统计信息
    print(f"\n创建内存中的小型训练子集: 共 {len(mini_train_df)} 条样本")
    print("标签分布:")
    print(mini_train_df['label'].value_counts())

    # 保存结果
    os.makedirs(output_dir, exist_ok=True)
    mini_train_df.to_csv(output_data_path, sep='\t', index=False, header=False)
    print("小型训练集已保存为 mini_train_set.csv")

    print('\n---')
    elapsed_time = time.time() - start_time_0
    print(f"预处理耗时: {elapsed_time} secs!")


