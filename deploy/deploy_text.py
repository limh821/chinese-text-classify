'''
    以下是一个文本识别和分类模型的部署脚本。

    包含如下功能：
    1、支持输入数据input_texts_xxx.csv带或不带user_uuid
    2、无论输入文件是否包含user_uuid列，输出文件中都会包含user_uuid列（如果不带则为空值）
    3、如果输入带user_uuid，则输出OUTPUT_FILENAME中也将带上与输入中text对应的user_uuid（程序会自动保证 text --> user_uuid的映射关系）
    4、包含一个基于THUCNews-mini 测试集的模型加载正确性的测试（模型参数文件也需要是基于THUCNews-mini 训练集训练的才行），输出准确率大于50%即表示模型正确加载（否则随机预测的准确率只有不到20%），
    如果不需要做模型加载正确性测试，可以设置TEST_MODEL_LOADING_WITH_THUCNEWS_MINI_DATASET_FLAG为False，同时设置SKIP_TEST_FLAG为True。

    by 李明华，2025-08-31.

'''


import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoConfig
from torch.utils.data import Dataset, DataLoader
import re
import os
import sys
import time
from tqdm import tqdm
import argparse
from datetime import datetime
from sklearn.metrics import accuracy_score, classification_report

# 检查GPU可用性
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")

# 训练好的模型的存放路径
MODEL_PATH = "./models/deployment_model"
# 测试集路径（用于验证模型加载是否正确）
TEST_SET_PATH = "./data/THUCNews-mini/test.txt"  # 修改为您的测试集路径
BATCH_SIZE = 16  # 根据GPU卡显存大小调整
# 输入与输出
INPUT_ROOT_PATH = "./data"
INPUT_FILENAME = 'input_texts_with_useruuid.csv'
INPUT = os.path.join(INPUT_ROOT_PATH, INPUT_FILENAME)
OUTPUT_ROOT_PATH = "./outputs"
OUTPUT_FILENAME = 'all_text_samples_info.csv'
OUTPUT = os.path.join(OUTPUT_ROOT_PATH, OUTPUT_FILENAME)

# 是否需要利用THUCNews-mini测试集检验模型是否正确且完整地加载（前提是模型必须要是使用THUCNews-mini训练集进行训练的才有效）
TEST_MODEL_LOADING_WITH_THUCNEWS_MINI_DATASET_FLAG = False
# 是否跳过模型测试
SKIP_TEST_FLAG = True

class TextClassificationDeployer:
    def __init__(self, model_path, device=DEVICE):
        """
        初始化文本分类部署器

        参数:
            model_path: 训练好的模型路径
            device: 运行设备
        """
        self.device = device
        self.model_path = model_path
        self.tokenizer = None
        self.model = None
        self.num_labels = None
        self.category_map = None

        # 初始化模型和分词器
        self._load_model_and_tokenizer()

    def _load_model_and_tokenizer(self):
        """加载模型和分词器（参考训练脚本的load_trained_lora_merged_model）"""
        print(f"正在加载模型和分词器从: {self.model_path}")
        start_time = time.time()

        # 检查模型路径是否存在
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"模型文件 {self.model_path} 不存在！")

        # 加载分词器
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path, model_max_length=256)

        # 加载配置
        config = AutoConfig.from_pretrained(self.model_path)
        self.num_labels = config.num_labels

        # 加载模型（参考训练脚本的方法）
        self.model = AutoModelForSequenceClassification.from_pretrained(
            self.model_path,
            config=config
        )

        self.model.to(self.device)
        self.model.eval()

        # 定义类别映射（根据您的实际类别修改）
        self.category_map = {
            0: "财经", 1: "房产", 2: "教育", 3: "科技",
            4: "军事", 5: "汽车", 6: "体育", 7: "游戏",
            8: "娱乐", 9: "政治"
        }

        # 验证模型是否正确加载（参考训练脚本的verify_model_loading）
        print("\n=== 模型加载验证 ===")

        # 检查分类头权重（判断是否与训练模型一致）
        if hasattr(self.model, 'classifier'):
            classifier_weight = self.model.classifier.weight.data
            classifier_norm = torch.norm(classifier_weight).item()
            print(f"分类头权重范数: {classifier_norm:.4f}")

        # # 检查LoRA层是否存在（非必需）
        # has_lora = any('lora' in name.lower() for name, _ in self.model.named_parameters())
        # print(f"包含LoRA层: {has_lora}")

        # 测试一条样本验证模型功能
        test_text = "苹果发布新款iPhone"
        test_pred, test_probs = self.predict_single_text(test_text)
        print(f"测试文本: '{test_text}'")
        print(f"预测结果: {test_pred}")
        print(f"各类别概率: {test_probs.round(4)}")

        load_time = time.time() - start_time
        print(f"模型加载验证完成，共 {self.num_labels} 个类别，耗时: {load_time:.2f} 秒")
        print(f"类别映射: {self.category_map}")

    def load_THUCNews_test(self, file_path):
        """ 加载CSV数据并分割为训练集和验证集 """
        # 读取Tab分隔的.txt文件
        test_df = pd.read_csv(file_path, sep='\t', header=None, nrows=1000, names=['text', 'label'])

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

    def test_model_loading(self, test_file_path, batch_size=32):
        """
        测试模型加载是否正确，使用带标签的测试集验证准确率
        这个函数可以快速注释掉，不影响主要部署功能
        """
        print("\n" + "=" * 60)
        print("开始测试模型加载正确性...")
        print("=" * 60)

        if not os.path.exists(test_file_path):
            print(f"测试集文件不存在: {test_file_path}")
            print("跳过模型测试验证")
            return

        # 加载测试集
        try:
            test_df = self.load_THUCNews_test(test_file_path)
            print(f"成功加载测试集: {len(test_df)} 条样本")
        except Exception as e:
            print(f"加载测试集失败: {e}")
            return

        # 清理测试数据：过滤掉NaN标签和空文本
        original_count = len(test_df)
        test_df = test_df.dropna(subset=['label'])  # 移除NaN标签
        test_df = test_df[test_df['label'].apply(lambda x: str(x).isdigit())]  # 确保标签是数字
        test_df['label'] = test_df['label'].astype(int)  # 转换为整数

        # 过滤空文本
        test_df = test_df[test_df['text'].notna() & (test_df['text'].str.len() > 0)]

        if len(test_df) < original_count:
            print(f"过滤掉 {original_count - len(test_df)} 条无效样本")

        if len(test_df) == 0:
            print("错误: 测试集没有有效样本")
            return

        print(f"有效测试样本: {len(test_df)} 条")

        # 创建测试数据集
        test_texts = test_df['text'].tolist()
        test_labels = test_df['label'].values

        # 批量预测
        predictions = self.predict_batch(test_texts, batch_size)
        pred_labels = [pred['pred_label'] for pred in predictions]

        # 检查预测结果是否有NaN
        valid_indices = []
        valid_true_labels = []
        valid_pred_labels = []

        for i, (true_label, pred_label) in enumerate(zip(test_labels, pred_labels)):
            if not np.isnan(pred_label) and not np.isnan(true_label):
                valid_indices.append(i)
                valid_true_labels.append(true_label)
                valid_pred_labels.append(pred_label)

        if len(valid_indices) < len(test_labels):
            print(f"过滤掉 {len(test_labels) - len(valid_indices)} 条无效预测")

        if len(valid_true_labels) == 0:
            print("错误: 没有有效的预测结果")
            return

        # 计算准确率
        accuracy = accuracy_score(valid_true_labels, valid_pred_labels)

        # 输出详细结果
        print(f"\n测试集准确率: {accuracy:.4f}")
        print(f"有效样本数: {len(valid_true_labels)}")

        # 输出各类别统计
        print("\n各类别分布 (真实标签):")
        unique_labels = np.unique(valid_true_labels)
        for label in unique_labels:
            count = sum(1 for true_label in valid_true_labels if true_label == label)
            print(f"  类别 {label}: {count} 条")

        # 检查准确率是否合理
        if accuracy < 0.5:
            print(f"⚠️  警告: 准确率较低 ({accuracy:.4f})，可能模型加载有问题")
            # 输出一些错误样本用于调试
            print("\n前5个错误预测样本:")
            error_count = 0
            for i in range(min(5, len(valid_true_labels))):
                if valid_true_labels[i] != valid_pred_labels[i]:
                    print(f"  文本: {test_texts[valid_indices[i]][:50]}...")
                    print(f"  真实: {valid_true_labels[i]}, 预测: {valid_pred_labels[i]}")
                    print("-" * 40)
                    error_count += 1
                    if error_count >= 3:
                        break
        else:
            print(f"✅ 模型加载验证通过，准确率: {accuracy:.4f}")

        print("=" * 60)
        print("模型测试完成")
        print("=" * 60)

        return accuracy

    def predict_single_text(self, text):
        """预测单条文本（参考训练脚本的predict_example）"""
        self.model.eval()
        inputs = self.tokenizer(
            text,
            add_special_tokens=True,
            max_length=256,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        ).to(self.device)

        with torch.no_grad():
            outputs = self.model(**inputs)
            probs = torch.nn.functional.softmax(outputs.logits, dim=1)
            pred = torch.argmax(probs, dim=1).item()

        return pred, probs.cpu().numpy()[0]

    def preprocess_text(self, text):
        """中文文本预处理"""
        if pd.isna(text) or text is None:
            return ""

        text = str(text)
        # 去除特殊字符和多余空格
        text = re.sub(r'[^\w\s]', '', text)
        text = re.sub(r'\s+', ' ', text).strip()
        return text

    def predict_batch(self, texts, batch_size=32):
        """
        批量预测文本分类（参考训练脚本的test_model_lora）

        参数:
            texts: 文本列表
            batch_size: 批处理大小

        返回:
            results: 预测结果列表，每个元素为字典
        """
        results = []

        # 创建数据集
        dataset = TextDataset(texts, self.tokenizer)
        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=0
        )

        self.model.eval()

        with torch.no_grad():
            for batch in tqdm(dataloader, desc="预测中"):
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)

                # 前向传播（参考训练脚本的test_model_lora）
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask
                )

                # 计算概率和预测结果
                logits = outputs.logits
                probs = torch.nn.functional.softmax(logits, dim=1)
                preds = torch.argmax(logits, dim=1)

                # 计算熵（不确定性度量）- 参考训练脚本
                entropy = -torch.sum(probs * torch.log(probs + 1e-10), dim=1)

                # 转换为numpy
                batch_probs = probs.cpu().numpy()
                batch_preds = preds.cpu().numpy()
                batch_entropy = entropy.cpu().numpy()

                # 处理每个样本的结果
                for i in range(len(batch_preds)):
                    result = {
                        'probabilities': batch_probs[i],
                        'pred_label': int(batch_preds[i]),  # 确保是整数
                        'pred_class_name': self.category_map.get(int(batch_preds[i]),
                                                                 f"未知类别_{int(batch_preds[i])}"),
                        'entropy': batch_entropy[i]
                    }
                    results.append(result)

        return results

    def detect_file_format(self, file_path):
        """
        检测文件格式和列信息
        返回: (has_user_uuid, separator)
        """
        try:
            # 先尝试tab分隔
            df_sample = pd.read_csv(file_path, sep='\t', nrows=5, encoding='utf-8')
            separator = '\t'
            separator_name = 'tab'
        except:
            try:
                # 尝试逗号分隔
                df_sample = pd.read_csv(file_path, sep=',', nrows=5, encoding='utf-8')
                separator = ','
                separator_name = '逗号'
            except Exception as e:
                print(f"无法识别文件分隔符: {e}")
                return False, None, None

        columns = df_sample.columns.tolist()
        has_user_uuid = 'user_uuid' in columns

        print(f"检测到文件分隔符: {separator_name}")
        print(f"文件列名: {columns}")
        print(f"包含user_uuid列: {has_user_uuid}")

        return has_user_uuid, separator, columns

    def process_input_file(self, input_file_path, output_file_path, batch_size=32):
        """
        处理输入文件并进行分类预测

        参数:
            input_file_path: 输入文件路径
            output_file_path: 输出文件路径
            batch_size: 批处理大小
        """
        print(f"正在读取输入文件: {input_file_path}")
        read_start_time = time.time()

        # 检测文件格式
        has_user_uuid, separator, columns = self.detect_file_format(input_file_path)
        if separator is None:
            return None

        # 读取输入文件
        try:
            df_input = pd.read_csv(input_file_path, sep=separator, encoding='utf-8')
        except Exception as e:
            print(f"读取文件失败: {e}")
            return None

        read_time = time.time() - read_start_time
        print(f"成功读取 {len(df_input)} 条数据，耗时: {read_time:.2f} 秒")

        # 检查必要的text列
        if 'text' not in df_input.columns:
            # 尝试找到包含文本的列
            text_columns = [col for col in df_input.columns if col.lower() in ['text', 'content', 'sentence', '文档']]
            if text_columns:
                df_input.rename(columns={text_columns[0]: 'text'}, inplace=True)
                print(f"重命名列 '{text_columns[0]}' 为 'text'")
            else:
                print("错误: 输入文件缺少text列或类似文本列")
                print(f"文件列名: {df_input.columns.tolist()}")
                return None

        # 预处理文本
        print("正在预处理文本...")
        preprocess_start_time = time.time()
        df_input['processed_text'] = df_input['text'].apply(self.preprocess_text)
        preprocess_time = time.time() - preprocess_start_time
        print(f"文本预处理完成，耗时: {preprocess_time:.2f} 秒")

        # 过滤空文本
        original_count = len(df_input)
        df_input = df_input[df_input['processed_text'].str.len() > 0]
        if len(df_input) < original_count:
            print(f"过滤掉 {original_count - len(df_input)} 条空文本")

        # 批量预测
        print("开始预测分类...")
        predict_start_time = time.time()
        texts = df_input['processed_text'].tolist()
        predictions = self.predict_batch(texts, batch_size)
        predict_time = time.time() - predict_start_time
        print(f"预测完成，耗时: {predict_time:.2f} 秒")

        # 合并预测结果到原始数据
        print("正在合并预测结果...")
        merge_start_time = time.time()

        # 创建概率列
        prob_columns = [f'prob_class_{i}' for i in range(self.num_labels)]

        # 提取概率值
        prob_values = [pred['probabilities'] for pred in predictions]
        prob_df = pd.DataFrame(prob_values, columns=prob_columns)

        # 提取其他预测信息
        pred_info = {
            'pred_label': [pred['pred_label'] for pred in predictions],
            'pred_class_name': [pred['pred_class_name'] for pred in predictions],
            'entropy': [pred['entropy'] for pred in predictions]
        }
        pred_df = pd.DataFrame(pred_info)

        # 构建结果DataFrame - 确保始终包含user_uuid列
        if has_user_uuid:
            # 包含user_uuid列，直接使用
            result_df = pd.concat([
                df_input[['user_uuid', 'text']].reset_index(drop=True),
                prob_df,
                pred_df
            ], axis=1)
        else:
            # 不包含user_uuid列，创建空的user_uuid列
            empty_user_uuid = pd.Series([None] * len(df_input), name='user_uuid')
            result_df = pd.concat([
                empty_user_uuid,
                df_input[['text']].reset_index(drop=True),
                prob_df,
                pred_df
            ], axis=1)

        # 确保列的顺序一致：user_uuid, text, 概率列, 预测信息列
        column_order = ['user_uuid', 'text'] + prob_columns + ['pred_label', 'pred_class_name', 'entropy']
        result_df = result_df[column_order]

        merge_time = time.time() - merge_start_time
        print(f"结果合并完成，耗时: {merge_time:.2f} 秒")

        # 保存结果
        print(f"正在保存结果到: {output_file_path}")
        save_start_time = time.time()
        result_df.to_csv(output_file_path, index=False, encoding='utf-8-sig')
        save_time = time.time() - save_start_time
        print(f"结果保存完成，耗时: {save_time:.2f} 秒")

        # 生成统计信息
        stats_start_time = time.time()
        self._generate_statistics(output_file_path, result_df, has_user_uuid)
        stats_time = time.time() - stats_start_time
        print(f"统计信息生成完成，耗时: {stats_time:.2f} 秒")

        return result_df, has_user_uuid

    def _generate_statistics(self, output_file_path, result_df, has_user_uuid):
        """生成统计信息"""
        print("\n" + "=" * 50)
        print("分类结果统计")
        print("=" * 50)

        total_count = len(result_df)
        print(f"总样本数: {total_count}")

        # 统计user_uuid情况
        if has_user_uuid:
            unique_users = result_df['user_uuid'].nunique()
            print(f"唯一用户数: {unique_users}")
            non_null_users = result_df['user_uuid'].notna().sum()
            print(f"有user_uuid的样本数: {non_null_users} ({non_null_users/total_count*100:.2f}%)")
        else:
            print("user_uuid列: 全部为空")

        # 各类别统计
        print("\n各类别分布:")
        class_distribution = result_df['pred_label'].value_counts().sort_index()
        for class_idx, count in class_distribution.items():
            class_name = self.category_map.get(class_idx, f"未知类别_{class_idx}")
            percentage = (count / total_count) * 100
            print(f"  类别 {class_idx}({class_name}): {count} 条 ({percentage:.2f}%)")

        # 熵统计
        avg_entropy = result_df['entropy'].mean()
        max_entropy = result_df['entropy'].max()
        min_entropy = result_df['entropy'].min()

        print(f"\n不确定性统计:")
        print(f"  平均熵: {avg_entropy:.4f}")
        print(f"  最大熵: {max_entropy:.4f}")
        print(f"  最小熵: {min_entropy:.4f}")

        # 高不确定性样本（熵大于平均熵+标准差）
        entropy_std = result_df['entropy'].std()
        high_entropy_threshold = avg_entropy + entropy_std
        high_entropy_count = len(result_df[result_df['entropy'] > high_entropy_threshold])

        print(
            f"  高不确定性样本数 (熵 > {high_entropy_threshold:.4f}): {high_entropy_count} 条 ({high_entropy_count / total_count * 100:.2f}%)")

        # 保存统计信息到文件
        output_dir = os.path.dirname(output_file_path)
        stats_file = os.path.join(output_dir, 'classification_statistics.txt')
        with open(stats_file, 'w', encoding='utf-8') as f:
            f.write("文本分类结果统计\n")
            f.write(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write("=" * 50 + "\n")
            f.write(f"总样本数: {total_count}\n")
            if has_user_uuid:
                f.write(f"唯一用户数: {unique_users}\n")
                f.write(f"有user_uuid的样本数: {non_null_users} ({non_null_users/total_count*100:.2f}%)\n")
            else:
                f.write("user_uuid列: 全部为空\n")
            f.write("\n")

            f.write("各类别分布:\n")
            for class_idx, count in class_distribution.items():
                class_name = self.category_map.get(class_idx, f"未知类别_{class_idx}")
                percentage = (count / total_count) * 100
                f.write(f"  类别 {class_idx}({class_name}): {count} 条 ({percentage:.2f}%)\n")

            f.write(f"\n不确定性统计:\n")
            f.write(f"  平均熵: {avg_entropy:.4f}\n")
            f.write(f"  最大熵: {max_entropy:.4f}\n")
            f.write(f"  最小熵: {min_entropy:.4f}\n")
            f.write(f"  高不确定性样本数: {high_entropy_count} 条 ({high_entropy_count / total_count * 100:.2f}%)\n")

        print(f"\n详细统计信息已保存到: {stats_file}")


class TextDataset(Dataset):
    """文本数据集类"""

    def __init__(self, texts, tokenizer, max_length=256):
        self.texts = texts
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])

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
            'attention_mask': encoding['attention_mask'].flatten()
        }


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='文本分类部署脚本')
    parser.add_argument('--model_path', type=str, default=MODEL_PATH,
                        help='训练好的模型路径')
    parser.add_argument('--input', type=str, default=INPUT,
                        help='输入文件路径 (CSV格式，包含user_uuid和text列)')
    parser.add_argument('--output', type=str, default=OUTPUT,
                        help='输出文件路径')
    parser.add_argument('--batch_size', type=int, default=BATCH_SIZE,
                        help='批处理大小，默认为32')
    parser.add_argument('--test_only', action='store_true', default=TEST_MODEL_LOADING_WITH_THUCNEWS_MINI_DATASET_FLAG,
                        help='仅测试模型，不进行部署预测')
    parser.add_argument('--skip_test', action='store_true', default=SKIP_TEST_FLAG,
                        help='跳过模型测试，直接进行部署')

    args = parser.parse_args()

    # 检查文件是否存在
    if not os.path.exists(args.model_path):
        print(f"错误: 模型路径不存在: {args.model_path}")
        return

    # 初始化部署器
    deployer = TextClassificationDeployer(args.model_path)

    # 测试模型加载正确性（除非指定跳过）
    if not args.skip_test:
        deployer.test_model_loading(TEST_SET_PATH, args.batch_size)
    else:
        print("跳过模型测试验证")

    # 如果只需要测试，不进行部署预测
    if args.test_only:
        print("测试完成，退出程序")
        return

    if not os.path.exists(args.input):
        print(f"错误: 输入文件不存在: {args.input}")
        return

    # 创建输出目录（如果不存在）
    output_dir = os.path.dirname(args.output)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # 记录总开始时间
    total_start_time = time.time()

    # 处理输入文件
    process_start_time = time.time()
    result = deployer.process_input_file(args.input, args.output, args.batch_size)
    process_time = time.time() - process_start_time

    if result is not None:
        result_df, has_user_uuid = result
        total_time = time.time() - total_start_time

        print("\n" + "=" * 60)
        print("处理完成!")
        print(f"总耗时: {total_time:.2f} 秒")
        print(f"处理耗时: {process_time:.2f} 秒")
        print(f"结果已保存到: {args.output}")
        print(f"输入文件包含user_uuid列: {has_user_uuid}")
        print("=" * 60)

        # 显示前几条结果
        print("\n前5条预测结果:")
        display_cols = ['user_uuid', 'text', 'pred_class_name', 'entropy']
        print(result_df[display_cols].head().to_string(index=False))
    else:
        print("处理失败!")

if __name__ == "__main__":
    main()

