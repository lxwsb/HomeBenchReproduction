import argparse
import json
import re
from collections import Counter
import os

# --- 路径设置 ---
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)

# 设置镜像
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

# --- 文本归一化函数 ---
def normalize_command(text):
    """
    去除参数名，统一格式。
    例如: set_brightness(brightness=70) -> set_brightness(70)
    """
    # 1. 去除参数名 (例如 'intensity=', 'brightness=', 'temperature=')
    # 匹配模式: 单词字符 + 等号
    text = re.sub(r'\b[a-zA-Z_]\w*=', '', text)
    return text

# --- 新增：核心解析函数 (正则提取) ---
def extract_commands(text):
    """
    从模型输出中稳健地提取指令。
    策略：
    1. 去除空格和换行。
    2. 使用正则寻找符合 '对象.方法(参数)' 格式的字符串 或 'error_input'。
    3. 忽略其他所有噪音文本。
    """
    if text is None: return []
    
    # 1. 基础清洗
    clean_text = text.replace(" ", "").replace("\n", "")
    
    # 2. 正则提取
    # 模式解释：
    # (?: ... ) : 非捕获组
    # [\w\.]+   : 匹配 room.device.method (字母数字下划线和点)
    # \(.*?\)   : 匹配括号及其中内容 (非贪婪)
    # |         : 或
    # error_input : 匹配错误标记
    pattern = r'(?:[\w\.]+\(.*?\)|error_input)'
    
    matches = re.findall(pattern, clean_text)
    
    return matches

def compute_accuracy(generated_texts, expected_texts, debug_limit=5):
    print("nums"   , len(generated_texts))
    correct_num = 0
    tp = 0
    all_pre = 0
    all_gold = 0
    res11 = []
    
    # 用于调试打印的计数器
    debug_count = 0
    
    for generated_text, expected_text in zip(generated_texts, expected_texts):
        
        original_generated = generated_text # 备份原始生成文本用于调试
        
        # --- 核心修改：使用正则提取代替简单的 split ---
        generated_list = extract_commands(generated_text)
        
        # 处理 expected (也做同样的清洗，防止格式不一致)
        # 这里的 expected_text 原始格式通常是逗号分隔的字符串
        expected_list = extract_commands(expected_text)

        # --- 应用归一化 ---
        generated_list = [normalize_command(x) for x in generated_list if x != ""]
        expected_list = [normalize_command(x) for x in expected_list if x != ""]
        
        generated_counter = Counter(generated_list)
        expected_counter = Counter(expected_list)
        
        if generated_counter == expected_counter:
            correct_num += 1
        else:
            res11.append({"generated": generated_list, "expected": expected_list})
            
            # --- 调试打印 ---
            if debug_count < debug_limit:
                print(f"\n[Mismatch Case {debug_count + 1}]")
                print(f"  Raw Generated : {repr(original_generated)}")
                print(f"  Parsed Result : {generated_list}")
                print(f"  Expected List : {expected_list}")
                debug_count += 1
        
        intersection = generated_counter & expected_counter

        tp += len(list(intersection.elements()))
        all_pre += len(generated_list)
        all_gold += len(expected_list)

    if len(generated_texts) == 0:
        print("No data to evaluate.")
        return []

    print("-" * 20)
    print("em:", correct_num / len(generated_texts))
    
    precision = tp / all_pre if all_pre > 0 else 0
    recall = tp / all_gold if all_gold > 0 else 0
    
    print("Precision:", precision)
    print("Recall:", recall)
    
    if precision + recall == 0:
        print("F1:", 0)
    else:
        print("F1:", 2 * precision * recall / (precision + recall))
    print("-" * 20)

    return res11


def dif_type(test_data):
    # 使用绝对路径读取原始数据集
    dataset_path = os.path.join(PROJECT_ROOT, "dataset", "test_data.jsonl")
    if not os.path.exists(dataset_path):
        print(f"Error: Dataset not found at {dataset_path}")
        return

    with open(dataset_path, "r") as f:
        data = f.readlines()
    
    normal_single = {"expected": [], "generated": []}
    unexist_single = {"expected": [], "generated": []}
    unexist_attribute_single = {"expected": [], "generated": []}
    unexist_device_single = {"expected": [], "generated": []}
    normal_multi = {"expected": [], "generated": []}
    mix_multi = {"expected": [], "generated": []}
    error_multi = {"expected": [], "generated": []}
    all_data = {"expected": [], "generated": []}
    
    print("test_data(results):", len(test_data))
    print("dataset(source):", len(data))
    
    # 确保长度一致，或者取较小值防止溢出
    min_len = min(len(test_data), len(data))
    
    for i in range(min_len):
        try:
            item_source = json.loads(data[i])
            item_result = test_data[i]
            
            # 简单的对齐容错
            if item_source["output"] != item_result["gold_output"]:
                pass 

            all_data["expected"].append(item_source["output"])
            all_data["generated"].append(item_result["generated_output"])
            
            item_type = item_source.get("type", "normal") # 默认 normal
            
            if item_type == "normal":
                normal_single["expected"].append(item_source["output"])
                normal_single["generated"].append(item_result["generated_output"])
            elif item_type == "unexist_device":
                unexist_device_single["expected"].append(item_source["output"])
                unexist_device_single["generated"].append(item_result["generated_output"])
                unexist_single["expected"].append(item_source["output"])
                unexist_single["generated"].append(item_result["generated_output"])
            elif item_type == "unexist_attribute":
                unexist_attribute_single["expected"].append(item_source["output"])
                unexist_attribute_single["generated"].append(item_result["generated_output"])
                unexist_single["expected"].append(item_source["output"])
                unexist_single["generated"].append(item_result["generated_output"])
            else:
                parts = item_type.split("_")
                if len(parts) > 1:
                    tmp = parts[1]
                    if tmp == "mix":
                        mix_multi["expected"].append(item_source["output"])
                        mix_multi["generated"].append(item_result["generated_output"])
                    elif tmp == "normal":
                        normal_multi["expected"].append(item_source["output"])
                        normal_multi["generated"].append(item_result["generated_output"])
                    else:
                        error_multi["expected"].append(item_source["output"])
                        error_multi["generated"].append(item_result["generated_output"])
                else:
                    # 兜底
                    error_multi["expected"].append(item_source["output"])
                    error_multi["generated"].append(item_result["generated_output"])
        except Exception as e:
            print(f"Error processing index {i}: {e}")
            continue

    output_dir = os.path.join(PROJECT_ROOT, "output")
    os.makedirs(output_dir, exist_ok=True)

    print("-" * 30)
    print("ALL DATA (Debug showing first 5 mismatches)")
    compute_accuracy(all_data["generated"], all_data["expected"], debug_limit=5)
    
    print("-" * 30)
    print("normal_single")
    compute_accuracy(normal_single["generated"], normal_single["expected"], debug_limit=0)
    
    print("-" * 30)
    print("unexist_single")
    compute_accuracy(unexist_single["generated"], unexist_single["expected"], debug_limit=0)
    
    # 写入错误分析文件
    with open(os.path.join(output_dir, "unexist_single_analysis.json"), "w") as f:
        pass 

    print("-" * 30)
    print("normal_multi")
    nm_error = compute_accuracy(normal_multi["generated"], normal_multi["expected"], debug_limit=0)
    with open(os.path.join(output_dir, "normal_multi_errors.json"), "w") as f:
        f.write(json.dumps(nm_error, indent=4))

    print("-" * 30)
    print("mix_multi")
    mm_error = compute_accuracy(mix_multi["generated"], mix_multi["expected"], debug_limit=0)
    with open(os.path.join(output_dir, "mix_multi_errors.json"), "w") as f:
        f.write(json.dumps(mm_error, indent=4))

    print("-" * 30)
    print("error_multi")
    em_error = compute_accuracy(error_multi["generated"], error_multi["expected"], debug_limit=0)
    with open(os.path.join(output_dir, "error_multi_errors.json"), "w") as f:
        f.write(json.dumps(em_error, indent=4))

    return None

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Model Evaluation Script for HomeBench.")
    parser.add_argument("--result_file", type=str, 
                        required=True, # 强制要求输入文件
                        help="Path to the model test result JSON file to evaluate.")
    
    args = parser.parse_args()

    print(f"Evaluating file: {args.result_file}")
    
    if not os.path.exists(args.result_file):
        print(f"Error: Result file not found: {args.result_file}")
        exit(1)

    try:
        with open(args.result_file, "r") as f:
            data = json.load(f)
    except json.JSONDecodeError as e:
        print(f"JSON Decode Error: {e}")
        print("Hint: The file might be in JSONL format but we expected JSON Array, or vice versa.")
        exit(1)

    test_data = []
    for item in data:
        # 兼容性处理：如果 item 已经是 dict，不需要再 loads
        if isinstance(item, str):
            try:
                item = json.loads(item)
            except:
                pass
        
        test_data.append({"generated_output": item["generated"], "gold_output": item["expected"]})
    
    dif_type(test_data)