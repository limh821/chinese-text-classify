'''
    实现各类别.csv训练集和测试集的分别拼接，支持包含表头和不包含表头的输入。

    使用注意：
    如有必要，请修改要处理的文件类型前缀：
        file_types = [
        ('mini_train_set_with_all_columns_class_', 'mini_train_set_with_all_columns_all_classes.csv'),
        ('mini_test_set_with_all_columns_class_', 'mini_test_set_with_all_columns_all_classes.csv'),
        ('mini_train_set_slim_class_', 'mini_train_set_slim_all_classes.csv'),
        ('mini_test_set_slim_class_', 'mini_test_set_slim_all_classes.csv')
    ]

    创建 by 李明华，2025-09-05.

'''

import os
import pandas as pd
import glob

# 如果输入文件包含表头，设置为True
HAS_HEADER = True


def simple_concatenate():
    """
    简化版拼接脚本
    """
    # 设置路径（根据实际情况修改）
    train_folder = '../data/preprocess/train_data_path/'
    test_folder = '../data/preprocess/test_data_path/'
    output_dir = '../data/preprocess/concatenated/'

    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)

    print(f"输入文件是否有表头: {HAS_HEADER}")
    print("=" * 60)

    # 定义要处理的文件类型前缀（如有必要，请修改这里）
    file_types = [
        ('mini_train_set_with_all_columns_', 'mini_train_set_with_all_columns_all_classes.csv'),  # 带所有列的完整的训练集
        ('mini_test_set_with_all_columns_', 'mini_test_set_with_all_columns_all_classes.csv'),  # 带所有列的完整的测试集
        ('mini_train_set_slim_', 'mini_train_set_slim_all_classes.csv'),  # 带所有列的完整的训练集
        ('mini_test_set_slim_', 'mini_test_set_slim_all_classes.csv')  # 带所有列的完整的训练集
    ]

    # 处理训练集和测试集
    for folder, folder_type in [(train_folder, 'train'), (test_folder, 'test')]:
        print(f"\n处理{folder_type}集文件夹: {folder}")

        for prefix, output_filename in file_types:
            if folder_type in prefix:  # 匹配对应的文件类型
                # 查找所有匹配的文件
                pattern = os.path.join(folder, f"{prefix}*.csv")
                files = glob.glob(pattern)

                if not files:
                    print(f"警告: 没有找到匹配 {pattern} 的文件")
                    continue

                print(f"找到 {len(files)} 个 {prefix}* 文件")

                # 读取并拼接文件
                dataframes = []
                total_rows = 0

                for i, file_path in enumerate(files):
                    try:
                        if HAS_HEADER:
                            # 有表头的情况：第一个文件读取表头，其他文件跳过表头
                            if i == 0:
                                df = pd.read_csv(file_path, sep='\t')
                                print(f"  {i + 1}. 读取(带表头): {os.path.basename(file_path)} ({len(df)} 行)")
                            else:
                                df = pd.read_csv(file_path, sep='\t', header=0)  # 跳过第一行表头
                                print(f"  {i + 1}. 读取(跳过表头): {os.path.basename(file_path)} ({len(df)} 行)")
                        else:
                            # 无表头的情况：直接读取所有行
                            df = pd.read_csv(file_path, sep='\t', header=None)
                            # 添加通用列名（可选）
                            df.columns = [f'col_{j}' for j in range(len(df.columns))]
                            print(f"  {i + 1}. 读取(无表头): {os.path.basename(file_path)} ({len(df)} 行)")

                        dataframes.append(df)
                        total_rows += len(df)

                    except Exception as e:
                        print(f"  读取 {file_path} 出错: {e}")

                if dataframes:
                    # 拼接DataFrame
                    result_df = pd.concat(dataframes, ignore_index=True)

                    # 保存结果
                    output_path = os.path.join(output_dir, output_filename)

                    if HAS_HEADER:
                        # 有表头：保存时包含表头
                        result_df.to_csv(output_path, sep='\t', index=False)
                        print(f"✅ 保存(带表头): {output_filename} ({len(result_df)} 行)")
                    else:
                        # 无表头：保存时不包含表头
                        result_df.to_csv(output_path, sep='\t', index=False, header=False)
                        print(f"✅ 保存(无表头): {output_filename} ({len(result_df)} 行)")

                    # 打印统计信息
                    print(f"  总处理行数: {total_rows}")
                    print(f"  拼接后行数: {len(result_df)}")
                    print(f"  列数: {len(result_df.columns)}")

    print("\n拼接完成！")


if __name__ == "__main__":
    simple_concatenate()



