'''
    实现文本训练集的快速分层抽样，支持百万级文本的秒级分层抽样。

    创建 by 李明华，2025-08-29.
    修改 by 李明华，2025-08-31，支持label在前text在后的数据格式

'''
import os.path
import pandas as pd
import numpy as np
from collections import defaultdict
import time

# 输入数据
input_data_path = '../data/THUCNews-mini-full/cnews.test.txt'
# 抽样数目
max_samples = 1000
# 输出目录
output_dir = '../data/preprocess/'
output_data_path = os.path.join(output_dir, 'mini_test_set.csv')

# 数据格式配置
# 如果数据格式是：第一列label，第二列text（含表头），设置为True
LABEL_FIRST_TEXT_SECOND = True
# 如果输入文件包含表头，设置为True
HAS_HEADER = False

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

    if HAS_HEADER:
        # 如果文件有表头，直接读取
        df = pd.read_csv(original_file, sep='\t')  # 假设新数据是Tab分隔

        # 检查列名是否正确
        if 'label' in df.columns and 'text' in df.columns:
            print("检测到标准列名: label, text")
        else:
            # 检查列名并重命名
            actual_columns = df.columns.tolist()
            print(f"实际列名: {actual_columns}")

            if len(actual_columns) >= 2:
                # 假设第一列是label，第二列是text
                if LABEL_FIRST_TEXT_SECOND:
                    if actual_columns[0] != 'label' or actual_columns[1] != 'text':
                        df = df.rename(columns={actual_columns[0]: 'label', actual_columns[1]: 'text'})
                        print(f"已重命名列: {actual_columns[0]} -> label, {actual_columns[1]} -> text")
                else:
                    if actual_columns[0] != 'text' or actual_columns[1] != 'label':
                        df = df.rename(columns={actual_columns[0]: 'text', actual_columns[1]: 'label'})
                        print(f"已重命名列: {actual_columns[0]} -> text, {actual_columns[1]} -> label")
    else:
        # 如果没有表头，根据数据格式设置列名
        if LABEL_FIRST_TEXT_SECOND:
            df = pd.read_csv(original_file, sep='\t', header=None, names=['label', 'text'])
        else:
            df = pd.read_csv(original_file, sep='\t', header=None, names=['text', 'label'])

    load_time = time.time() - start_time
    print(f"原始训练集大小：{len(df)} 条，读取耗时：{load_time:.2f}秒")
    print(f"开始分层抽样生成大小为 {max_samples} 条的 mini train set")

    # 检查数据格式
    print(f"数据前5行:")
    print(df.head())
    print(f"标签类型: {df['label'].dtype}")
    print(f"标签唯一值: {df['label'].unique()[:10]}")  # 显示前10个唯一标签
    print(f"标签分布:\n{df['label'].value_counts()}")

    # 使用优化后的分层抽样函数
    start_sample_time = time.time()
    mini_df = optimized_stratified_sample(df, 'label', max_samples, random_state)
    sample_time = time.time() - start_sample_time

    # 打印统计信息
    print(f"\n创建内存中的小型训练子集完成: 共 {len(mini_df)} 条样本，抽样耗时：{sample_time:.2f}秒")
    print("抽样后标签分布:")
    label_distribution = mini_df['label'].value_counts()
    print(label_distribution)

    # 检查每个类别的样本数是否相等
    unique_counts = label_distribution.unique()
    if len(unique_counts) == 1:
        print(f"✅ 所有类别样本数相等: {unique_counts[0]} 条/类别")
    else:
        print(f"⚠️  类别样本数不完全相等: {unique_counts.min()} - {unique_counts.max()} 条/类别")

    print(f"抽样前后比例一致性检查:")
    original_prop = df['label'].value_counts(normalize=True).sort_index()
    sampled_prop = mini_df['label'].value_counts(normalize=True).sort_index()

    for label in original_prop.index:
        if label in sampled_prop:
            diff = abs(original_prop[label] - sampled_prop[label])
            count_diff = abs(len(df[df['label'] == label]) - len(mini_df[mini_df['label'] == label]))
            print(f"  标签 {label}: 原始 {original_prop[label]:.3f} -> 抽样 {sampled_prop[label]:.3f} (差异: {diff:.3f}, 数量差: {count_diff})")

    return mini_df

def optimized_stratified_sample(df, label_col, n_samples, random_state=None):
    """
    高效的分层抽样函数，专门优化用于文本数据
    确保每个类别的样本数尽可能相等
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
    for class_val in unique_classes:
        class_indices[class_val] = np.where(all_labels == class_val)[0]

    # 3. 计算每个类别的抽样数量（确保尽可能相等）
    num_classes = len(unique_classes)
    base_samples_per_class = n_samples // num_classes
    remainder = n_samples % num_classes

    sample_sizes = {}
    for i, class_val in enumerate(unique_classes):
        # 基本样本数 + 如果有余数，前几个类别多分一个
        sample_sizes[class_val] = base_samples_per_class + (1 if i < remainder else 0)

        # 确保不超过该类别的实际样本数
        available_samples = len(class_indices[class_val])
        if sample_sizes[class_val] > available_samples:
            sample_sizes[class_val] = available_samples
            print(f"警告: 类别 {class_val} 样本不足，只能抽取 {available_samples} 条")

    # 4. 收集抽样索引（使用numpy随机选择）
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

    # 5. 如果样本不足，补充随机样本（确保总数正确）
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

    # 6. 创建结果DataFrame并打乱顺序
    result_df = df.iloc[sampled_indices].copy()
    result_df = result_df.sample(frac=1, random_state=random_state).reset_index(drop=True)

    return result_df

# 使用示例
if __name__ == "__main__":
    start_time_0 = time.time()

    # 测试高效版本
    mini_train_df = create_mini_dataset_fast(input_data_path, max_samples=max_samples, random_state=42)

    # 打印统计信息
    print(f"\n最终抽样结果: 共 {len(mini_train_df)} 条样本")
    final_distribution = mini_train_df['label'].value_counts()
    print("最终标签分布:")
    print(final_distribution)

    # 检查均匀性
    counts = final_distribution.values
    if np.all(counts == counts[0]):
        print("✅ 所有类别样本数完全相等!")
    else:
        print(f"⚠️  类别样本数差异: 最小 {counts.min()}, 最大 {counts.max()}, 差异 {counts.max() - counts.min()}")

    # 保存结果 - 保持原始的中文标签
    os.makedirs(output_dir, exist_ok=True)

    # 根据数据格式决定保存方式
    if LABEL_FIRST_TEXT_SECOND:
        # 保存为label在前，text在后的格式
        mini_train_df[['label', 'text']].to_csv(output_data_path, sep='\t', index=False, header=False)
        print("小型训练集已保存为 CSV 格式 (label, text)，包含表头")
    else:
        # 保存为text在前，label在后的格式
        mini_train_df[['text', 'label']].to_csv(output_data_path, sep='\t', index=False, header=False)
        print("小型训练集已保存为 TSV 格式 (text, label)，无表头")

    print('\n---')
    elapsed_time = time.time() - start_time_0
    print(f"预处理耗时: {elapsed_time:.2f} secs!")

