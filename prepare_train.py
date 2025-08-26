'''
    做文本分类模型的微调训练，需要准备训练集和测试集。数据集中有数量不等的10个类别的文本，标签分别是0-9，text_dataset.csv内容和格式如下：
        中华女子学院：本科层次仅1专业招男生	3
        两天价网站背后重重迷雾：做个网站究竟要多少钱	4
        东5环海棠公社230-290平2居准现房98折优惠	1
        卡佩罗：告诉你德国脚生猛的原因 不希望英德战踢点球	7
        ...
    下面python脚本对text_dataset.csv做处理，将里面的样本按照train_ratio = 0.8（这个train_ratio比例代表text_dataset.csv中数据总量的0.8作为训练集，其余作为测试集，
    train_ratio可以随意指定）分成训练集和测试集，同时需要保证测试集和训练集中各类标签的分布一致（即各类标签的比例要保持和原始text_dataset.csv中的比例一致）。

    by 李明华，2025-08-26.

'''

import os
import pandas as pd
from sklearn.model_selection import train_test_split



def split_dataset_with_stratification(input_file, train_file, test_file, train_ratio=0.8, random_state=42):
    """
    按指定比例分割数据集，保持类别分布一致

    参数:
    input_file: 输入CSV文件路径
    train_ratio: 训练集比例
    random_state: 随机种子
    """

    # 读取数据，指定分隔符为制表符
    df = pd.read_csv(input_file, sep='\t', header=None, names=['text', 'label'], on_bad_lines='warn')

    print(f"原始数据集信息:")
    print(f"总样本数: {len(df)}")
    print(f"各类别分布:")
    print(df['label'].value_counts().sort_index())

    # 按标签分层分割数据集
    train_df, test_df = train_test_split(
        df,
        train_size=train_ratio,
        random_state=random_state,
        stratify=df['label'],
        shuffle=True
    )

    # 保存训练集和测试集
    train_df.to_csv(train_file, index=False, header=False)
    test_df.to_csv(test_file, index=False, header=False)

    print(f"\n分割结果:")
    print(f"训练集样本数: {len(train_df)} ({len(train_df) / len(df) * 100:.1f}%)")
    print(f"测试集样本数: {len(test_df)} ({len(test_df) / len(df) * 100:.1f}%)")

    print(f"\n训练集类别分布:")
    print(train_df['label'].value_counts().sort_index())

    print(f"\n测试集类别分布:")
    print(test_df['label'].value_counts().sort_index())

    print(f"\n文件已保存:")
    print(f"训练集: {train_file}")
    print(f"测试集: {test_file}")

    return train_df, test_df


if __name__ == "__main__":
    # 参数设置
    input_dir = './data/THUCNews-mini/'
    input_file_name = 'train.txt'
    train_ratio = 0.8
    random_state = 42
    input_file = os.path.join(input_dir, input_file_name)

    # 结果输出
    output_dir = './outputs/preprocessed_data'
    os.makedirs(output_dir, exist_ok=True)
    train_file = os.path.join(output_dir, 'train_dataset.csv')
    test_file = os.path.join(output_dir, 'test_dataset.csv')

    # 执行数据集分割
    train_data, test_data = split_dataset_with_stratification(
        input_file,
        train_file,
        test_file,
        train_ratio,
        random_state
    )

