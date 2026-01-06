import json
import argparse
import nltk
from nltk.metrics import precision, recall, f_measure
import numpy as np
import jieba
import re
from nltk.translate import meteor_score
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from functools import partial
import multiprocessing
from tqdm import tqdm


def contain_chinese_string(text):
    """
    使用正则表达式检查字符串中是否包含中文字符
    """
    chinese_pattern = re.compile(r'[\u4e00-\u9fa5]')
    return bool(chinese_pattern.search(text))

def cal_per_metrics(image, pred, gt):
    """
    比较预测文本和真实文本，计算各种评估指标
    计算指标包括：BLEU、METEOR、F-measure、Precision、Recall、编辑距离
    适用于中英文混合文本的评估
    传入参数：
    - pred: 预测文本字符串
    - gt: 真实文本字符串
    返回值：
    - metrics: 包含各项评估指标的字典
    """

    metrics = {}
    metrics["image"] = image

    # 根据文本内容选择分词方式
    if contain_chinese_string(gt) or contain_chinese_string(pred):
        reference = jieba.lcut(gt)
        hypothesis = jieba.lcut(pred)
    else:
        reference = gt.split()
        hypothesis = pred.split()

    # 计算各项指标
    metrics["bleu"] = nltk.translate.bleu([reference], hypothesis)
    metrics["meteor"] = meteor_score.meteor_score([reference], hypothesis)

    reference = set(reference)
    hypothesis = set(hypothesis)
    
    metrics["f_measure"] = f_measure(reference, hypothesis)
    metrics["precision"] = precision(reference, hypothesis)
    metrics["recall"] = recall(reference, hypothesis)
    metrics["edit_dist"] = nltk.edit_distance(pred, gt) / max(len(pred), len(gt))
    print(f"Image: {image}, BLEU: {metrics['bleu']:.4f}, METEOR: {metrics['meteor']:.4f}, F-measure: {metrics['f_measure']:.4f}, Precision: {metrics['precision']:.4f}, Recall: {metrics['recall']:.4f}, Edit Distance: {metrics['edit_dist']:.4f}")
    return metrics

def eval(predicts):
    """
    对预测结果文件进行评估，计算整体的评估指标
    批量评估OCR预测结果并计算平均指标
    预测结果格式为：
    [
        {
            "label": "预测文本",
            "answer": "真实标签"
        },
        ...
    ]
    输出整体评估指标的平均值
    """
    
    eval_results = []
    for pred in predicts:
        ans = cal_per_metrics(pred["image"], pred["raw_ocr_text"], pred["gt_text"])
        eval_results.append(ans)
    
    mean_dict = {}
    mean_dict["eval question num"] = len(eval_results)
    
    
    
    # 按照img名称重新排序结果
    eval_results = sorted(
        eval_results,
        key=lambda x: int(x["image"].split("_")[-1].split(".")[0])
    )
    
    
    mean_dict = {}
    mean_dict["eval question num"] = len(eval_results)
    
    # ✅ 只初始化数值类型的指标
    for k, v in eval_results[0].items():
        if k != "image":
            mean_dict[k] = 0.0

    # ✅ 累加时排除 "image" 字段
    for each in eval_results:
        for k, v in each.items():
            if k != "image":
                mean_dict[k] += v
    
    # 计算平均值
    for k in list(mean_dict.keys()):
        if k == "eval question num":
            continue
        mean_dict[k] /= len(eval_results)
    
    # 打印结果
    print("\n" + "="*60)
    print("Evaluation Results:")
    print("="*60)
    for k, v in mean_dict.items():
        if k == "eval question num":
            print(f"{k}: {int(v)}")
        else:
            print(f"{k}: {v:.4f}")
    print("="*60)
        
    return eval_results , mean_dict

def main(predict_file, output_file):    
    with open(predict_file, "r", encoding="utf-8") as f:
        predict_data = json.load(f)
        
    results, mean_dict = eval(predict_data)
    # 将image相同的评估结果保存到原始预测结果中
    for item in tqdm(predict_data):
        image = item["image"]
        for res in results:
            if res["image"] == image:
                item.update(res)
                break
    
    # 额外添加整体评估结果
    predict_data.append({"overall_metrics": mean_dict})

    # 保存结果
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(predict_data, f, ensure_ascii=False, indent=4)
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--predict_file", type=str, required=True, help="Path to the JSON file containing OCR predictions.")
    parser.add_argument("--output_file", type=str, required=True, help="Path to save the evaluation results.")
    args = parser.parse_args()
    
    main(args.predict_file, args.output_file)
