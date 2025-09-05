'''
    实现文本训练集的快速分层抽样，支持百万级文本的秒级分层抽样。

    创建 by 李明华，2025-08-29.
    修改 by 李明华，2025-08-31，支持label在前text在后的数据格式
    修改 by 李明华，2025-09-01，增加测试集与训练集去重功能

    功能调整如下：
    1、能自动读取带任意表头的数据，列与列之间以Tab键分隔
    2、LABEL_FIRST_TEXT_SECOND 设置去掉，改为按实际表头自动读取。
    3、HAS_HEADER保留，需要人工指定输入是否带表头，是则按照实际表头读取；否则需要在custom_column_names中依次指定每列的表头名称。
    4、读取后，将content_tag列表头重命名为label，content列表头重命名为text，其他字段名字保持不变
    5、自动检测独立label类别数目（即num_classes）和每个类别class_i的样本总数目 total_sample_class_i
    6、支持一次性生成抽样训练集和抽样测试集。先抽样生成训练集，再抽样生成测试集，测试集里的text内容不能和训练集中的text内容重复。同时支持指定抽样测试集和训练集的比例train_test_split_ratio。
    7、可以支持智能分层抽样，抽样后每个类别的样本数目应该相等且等于max_samples/num_classes，如果某个类别（类别i）的总样本数目为total_sample_class_i小于max_samples/num_classes, 则该类别抽样数目自动调整为 min(math.floor(max_samples/num_classes), total_sample_class_i)
    8、输出目录设置为 output_dir = '../data/preprocess/'，抽样后的测试集和训练集数据分别存放在output_dir下面的train_data_path和test_data_path文件夹下
    9、去重功能保留，但增加去重设置变量do_remove_duplicates_flag (默认值为True)，为True时候才做去重，为False则不做。为True时，如果没有发现可以用于去重参考的train_data_path下的抽样后的训练集数据文件，则跳过去重功能并输出警告信息，直接输出测试集抽样结果。
    10、分别输出两种格式的训练集和测试集到train_data_path和test_data_path，带所有原始字段和列的mini_train_set_with_all_columns.csv, mini_test_set_with_all_columns.csv和仅仅带有label和text字段的mini_train_set_slim.csv, mini_test_set_slim.csv
    11、打印输出输入文件是否带表头，所有表头的名字，打印3条原始数据以用于检查读取正确与否，抽样前后总样本和各类标签样本的数目，打印3条抽样后的数据以用于检查内容是否正确，等信息。

    修改 by 李明华，2025-09-03，完善上述抽样功能调整，能一次性生成训练集和测试集，并智能处理自定义表头。
    修改 by 李明华，2025-09-03，增加原始数据中text内容的去重，先去重再进行抽样，确保不会因数据重复导致最终生成的训练集和测试集的比例和tran_test_ratio要求的差太远。

'''

import os.path
import pandas as pd
import numpy as np
from collections import defaultdict
import time
import math

# 输入数据路径
input_data_path = '../data/THUCNews-mini-full/cnews.test.txt'
# 输出目录
output_dir = '../data/preprocess/'
# 训练集和测试集输出路径
train_data_path = os.path.join(output_dir, 'train_data_path')
test_data_path = os.path.join(output_dir, 'test_data_path')
# 输出文件名
classname = 'all'
output_train_dataset_filename_full = 'mini_train_set_with_all_columns_' + classname + '.csv'
output_test_dataset_filename_full = 'mini_test_set_with_all_columns_' + classname + '.csv'
output_train_dataset_filename_slim = 'mini_train_set_slim_' + classname + '.csv'
output_test_dataset_filename_slim = 'mini_test_set_slim_' + classname + '.csv'

# 数据格式配置
# 如果输入文件包含表头，设置为True
HAS_HEADER = False
# 如果没有表头，需要在此指定列名（按实际顺序）
custom_column_names = ['content_tag', 'content', 'other_column1', 'other_column2']  # 根据实际情况修改

# 抽样配置
max_samples = 1000  # 总样本数
train_test_split_ratio = 0.8  # 训练集比例
random_state = 42  # 随机种子

# 去重设置
do_remove_duplicates_flag = True  # 是否进行去重

# 重命名映射（将content_tag重命名为label，content重命名为text）
column_rename_map = {'content_tag': 'label', 'content': 'text'}


def load_data(file_path, has_header, custom_names=None):
    """读取数据文件，支持带表头或不带表头"""
    print(f"正在读取文件: {file_path}")

    if has_header:
        # 读取带表头的数据
        df = pd.read_csv(file_path, sep='\t')
        print(f"检测到表头: {df.columns.tolist()}")
    else:
        # 读取不带表头的数据，使用自定义列名
        if custom_names is None:
            raise ValueError("当HAS_HEADER=False时，必须提供custom_column_names")
        df = pd.read_csv(file_path, sep='\t', header=None, names=custom_names)
        print(f"使用自定义列名: {df.columns.tolist()}")

    # 重命名列
    df = df.rename(columns=column_rename_map)
    print(f"重命名后列名: {df.columns.tolist()}")

    return df


def print_data_info(df, title="数据信息"):
    """打印数据的基本信息"""
    print(f"\n=== {title} ===")
    print(f"总样本数: {len(df)}")
    print(f"列名: {df.columns.tolist()}")

    if 'label' in df.columns:
        num_classes = df['label'].nunique()
        print(f"独立标签类别数目 (num_classes): {num_classes}")
        print("标签分布:")
        label_distribution = df['label'].value_counts()
        print(label_distribution)

        # 打印每个类别的样本总数
        print("\n每个类别的样本总数:")
        for label, count in label_distribution.items():
            print(f"  类别 '{label}': {count} 条样本")

    print("\n前3条数据:")
    for i, row in df.head(3).iterrows():
        print(f"  第{i + 1}条: {row.to_dict()}")

    return num_classes if 'label' in df.columns else 0


def remove_duplicate_texts(df, reference_texts):
    """移除与参考集中重复的文本"""
    if not reference_texts:
        return df, 0

    original_count = len(df)

    # 创建文本的清洗版本用于比较
    df['cleaned_text'] = df['text'].astype(str).str.strip()

    # 找出重复的文本
    duplicate_mask = df['cleaned_text'].isin(reference_texts)
    duplicate_count = duplicate_mask.sum()

    if duplicate_count > 0:
        print(f"发现 {duplicate_count} 条重复文本")
        print("重复文本示例:")
        duplicate_samples = df[duplicate_mask].head(3)
        for _, row in duplicate_samples.iterrows():
            print(f"  标签: {row['label']}, 文本: {row['text'][:50]}...")

        # 移除重复文本
        df = df[~duplicate_mask].copy()
        print(f"移除重复文本后剩余 {len(df)} 条数据")
    else:
        print("✅ 未发现重复文本")

    # 移除临时列
    df = df.drop(columns=['cleaned_text'])
    return df, duplicate_count


def stratified_sample(df, label_col, n_samples, random_state=None):
    """
    分层抽样函数，确保每个类别的样本数尽可能相等
    """
    if n_samples >= len(df):
        return df.copy()

    if random_state is not None:
        np.random.seed(random_state)

    # 获取所有标签
    all_labels = df[label_col].values
    unique_classes, class_counts = np.unique(all_labels, return_counts=True)
    class_indices = {}

    # 构建类别索引映射
    for class_val in unique_classes:
        class_indices[class_val] = np.where(all_labels == class_val)[0]

    # 计算每个类别的抽样数量
    num_classes = len(unique_classes)
    target_samples_per_class = n_samples // num_classes
    remainder = n_samples % num_classes

    sample_sizes = {}
    for i, class_val in enumerate(unique_classes):
        # 基本样本数 + 余数分配
        sample_size = target_samples_per_class + (1 if i < remainder else 0)

        # 确保不超过该类别的实际样本数
        available_samples = len(class_indices[class_val])
        sample_sizes[class_val] = min(sample_size, available_samples)

        if sample_size > available_samples:
            print(f"警告: 类别 '{class_val}' 样本不足，只能抽取 {available_samples} 条")

    # 收集抽样索引
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

    # 如果样本不足，补充随机样本
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

    # 创建结果DataFrame并打乱顺序
    result_df = df.iloc[sampled_indices].copy()
    result_df = result_df.sample(frac=1, random_state=random_state).reset_index(drop=True)

    return result_df

def remove_duplicate_texts_within_dataset(df):
    """移除数据集内部的重复文本，相同text的记录只保留第一条"""
    original_count = len(df)

    # 创建文本的清洗版本用于比较
    df['cleaned_text'] = df['text'].astype(str).str.strip()

    # 找出重复的文本（保留第一条，标记后续重复的）
    duplicate_mask = df.duplicated(subset=['cleaned_text'], keep='first')
    duplicate_count = duplicate_mask.sum()

    if duplicate_count > 0:
        print(f"发现数据集内部 {duplicate_count} 条重复文本")
        print("重复文本示例:")
        duplicate_samples = df[duplicate_mask].head(3)
        for _, row in duplicate_samples.iterrows():
            print(f"  标签: {row['label']}, 文本: {row['text'][:50]}...")

        # 移除重复文本（只保留第一条）
        df = df[~duplicate_mask].copy()
        print(f"移除内部重复文本后剩余 {len(df)} 条数据")
    else:
        print("✅ 未发现数据集内部重复文本")

    # 移除临时列
    df = df.drop(columns=['cleaned_text'])
    return df, duplicate_count

def create_train_test_datasets(input_file, max_samples, split_ratio, random_state, do_remove_duplicates=True):
    """
    创建训练集和测试集
    """
    # 读取原始数据
    df = load_data(input_file, HAS_HEADER, custom_column_names)
    num_classes = print_data_info(df, "原始数据信息")

    # text内容去重
    print("\n=== 原始数据集text内容去重 ===")
    df, internal_duplicate_count = remove_duplicate_texts_within_dataset(df)
    if len(df) == 0:
        print("错误: 去重后数据集为空！")
        return pd.DataFrame(), pd.DataFrame(), internal_duplicate_count
    else:
        # 更新一下实际抽样数，保证不要超过去重后的总样本数目
        max_samples = min(max_samples, len(df))

    # 计算训练集和测试集大小
    train_size = int(max_samples * split_ratio)
    test_size = max_samples - train_size

    print(f"\n计划抽样: 总样本 {max_samples} (训练集 {train_size}, 测试集 {test_size})")

    # 分层抽样训练集
    print("\n=== 抽样训练集 ===")
    train_df = stratified_sample(df, 'label', train_size, random_state)
    print_data_info(train_df, "训练集抽样结果")

    # 保存训练集文本用于去重
    train_texts = set(train_df['text'].astype(str).str.strip())

    # 从剩余数据中抽样测试集
    print("\n=== 抽样测试集 ===")
    # 获取训练集索引
    train_indices = set(train_df.index)
    # 剩余数据
    remaining_df = df[~df.index.isin(train_indices)].copy()

    print(f"剩余数据量: {len(remaining_df)}")

    # 如果需要去重且训练集不为空
    if do_remove_duplicates and train_texts:
        print("进行测试集去重检查...")
        remaining_df, duplicate_count = remove_duplicate_texts(remaining_df, train_texts)
        print(f"去重后剩余数据量: {len(remaining_df)}")
    else:
        print("跳过测试集去重检查")
        duplicate_count = 0

    # 抽样测试集
    if len(remaining_df) < test_size:
        print(f"警告: 剩余数据不足，测试集只能抽取 {len(remaining_df)} 条")
        test_size = len(remaining_df)

    test_df = stratified_sample(remaining_df, 'label', test_size, random_state + 1)
    print_data_info(test_df, "测试集抽样结果")

    return train_df, test_df, duplicate_count


def save_datasets(train_df, test_df, output_dir):
    """保存训练集和测试集"""
    # 创建输出目录
    train_dir = os.path.join(output_dir, 'train_data_path')
    test_dir = os.path.join(output_dir, 'test_data_path')
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)

    # 保存完整数据集（包含所有列）
    train_full_path = os.path.join(train_dir, output_train_dataset_filename_full)
    test_full_path = os.path.join(test_dir, output_test_dataset_filename_full)

    train_df.to_csv(train_full_path, sep='\t', index=False)
    test_df.to_csv(test_full_path, sep='\t', index=False)

    print(f"完整训练集已保存: {train_full_path}")
    print(f"完整测试集已保存: {test_full_path}")

    # 保存精简数据集（只包含label和text）
    if 'label' in train_df.columns and 'text' in train_df.columns:
        train_slim_path = os.path.join(train_dir, output_train_dataset_filename_slim)
        test_slim_path = os.path.join(test_dir, output_test_dataset_filename_slim)

        # # slim格式的输出不带表头
        # train_df[['label', 'text']].to_csv(train_slim_path, sep='\t', index=False, header=False)
        # test_df[['label', 'text']].to_csv(test_slim_path, sep='\t', index=False, header=False)
        # 带表头的slim格式输出
        train_df[['label', 'text']].to_csv(train_slim_path, sep='\t', index=False)
        test_df[['label', 'text']].to_csv(test_slim_path, sep='\t', index=False)

        print(f"精简训练集已保存: {train_slim_path}")
        print(f"精简测试集已保存: {test_slim_path}")
    else:
        print("警告: 数据中缺少label或text列，无法保存精简数据集")


def verify_no_duplicates(train_df, test_df):
    """验证训练集和测试集没有重复文本"""
    train_texts = set(train_df['text'].astype(str).str.strip())
    test_texts = set(test_df['text'].astype(str).str.strip())

    duplicates = test_texts & train_texts

    if duplicates:
        print(f"❌ 验证失败: 发现 {len(duplicates)} 条重复文本")
        print("重复文本示例:")
        for i, text in enumerate(list(duplicates)[:3]):
            print(f"  {i + 1}. {text[:50]}...")
    else:
        print("✅ 验证成功: 训练集与测试集没有重复文本")


# 主程序
if __name__ == "__main__":
    start_time = time.time()

    print("=== 文本分类数据预处理脚本 ===")
    print(f"输入文件: {input_data_path}")
    print(f"是否有表头: {HAS_HEADER}")
    if not HAS_HEADER:
        print(f"自定义列名: {custom_column_names}")
    print(f"重命名映射: {column_rename_map}")
    print(f"总样本数: {max_samples}")
    print(f"训练集比例: {train_test_split_ratio}")
    print(f"去重设置: {do_remove_duplicates_flag}")

    # 创建训练集和测试集
    train_df, test_df, duplicate_count = create_train_test_datasets(
        input_data_path, max_samples, train_test_split_ratio,
        random_state, do_remove_duplicates_flag
    )
    print(f"总共移除 {duplicate_count} 条重复文本")

    # 保存数据集
    print("\n=== 保存数据集 ===")
    save_datasets(train_df, test_df, output_dir)

    # 验证去重结果
    print("\n=== 去重验证 ===")
    verify_no_duplicates(train_df, test_df)

    # 打印最终统计
    print("\n=== 最终统计 ===")
    print(f"训练集大小: {len(train_df)}")
    print(f"测试集大小: {len(test_df)}")
    print(f"总样本数: {len(train_df) + len(test_df)}")

    elapsed_time = time.time() - start_time
    print(f"\n预处理总耗时: {elapsed_time:.2f} 秒")

