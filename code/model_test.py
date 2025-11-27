import argparse
import torch
import os
import json
import re
import time
import glob
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler
import torch.distributed as dist
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig
from transformers import Qwen2TokenizerFast, PreTrainedTokenizerFast
# 新增: 导入 PeftModel 用于加载 LoRA 权重
from peft import PeftModel

# --- 路径设置 ---
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)

# 设置镜像
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

# --- 辅助函数 (保持不变) ---
def chang_json2str(state, methods):
    state_str = ""
    for room in state.keys():
        state_str += room + ":\n"
        if room == "VacuumRobot":
            if isinstance(state[room], dict):
                state_str += "  state: " + str(state[room].get("state", "N/A")) + "\n"
                if "attributes" in state[room]:
                    for attribute in state[room]["attributes"].keys():
                        val = state[room]["attributes"][attribute]
                        state_str += "  " + attribute + ": " + str(val.get("value", "N/A"))
                        if "options" in val:
                            state_str += " (options" + str(val["options"]) + ")\n"
                        elif "lowest" in val:
                            state_str += " (range: " + str(val.get("lowest")) + " - " + str(val.get("highest")) + ")\n"
                        else:
                            state_str += "\n"
        else:
            for device in state[room].keys():
                if device == "room_name": continue
                device_obj = state[room][device]
                state_str += "  " + device + "\n"
                state_str += "    state: " + str(device_obj.get("state", "N/A")) + "\n"
                
                if "attributes" in device_obj:
                    for attribute in device_obj["attributes"].keys():
                        val = device_obj["attributes"][attribute]
                        state_str += "    " + attribute + ": " + str(val.get("value", "N/A"))
                        if "options" in val:
                            state_str += " (options" + str(val["options"]) + ")\n"
                        elif "lowest" in val:
                            state_str += " (range: " + str(val.get("lowest")) + " - " + str(val.get("highest")) + ")\n"
                        else:
                            state_str += "\n"

    method_str = ""
    for method in methods:
        room_prefix = method["room_name"] + "." if method["room_name"] != "None" else ""
        method_str += f"{room_prefix}{method['device_name']}.{method['operation']}("
        
        if len(method["parameters"]) > 0:
            params = [f"{p['name']}:{p['type']}" for p in method["parameters"]]
            method_str += ",".join(params)
        method_str += ");"
    return state_str, method_str

# --- Dataset 类 (保持不变) ---
class no_few_shot_home_assistant_dataset(Dataset):
    def __init__(self, tokenizer, use_rag=False):
        self.tokenizer = tokenizer
        dataset_dir = os.path.join(PROJECT_ROOT, "dataset")
        code_dir = os.path.join(PROJECT_ROOT, "code")
        
        if use_rag:
            with open(os.path.join(dataset_dir, "rag_test_data.json"), "r") as f:
                self.data = json.loads(f.read())
        else:
            with open(os.path.join(dataset_dir, "test_data.jsonl"), "r") as f:
                lines = f.readlines()
            with open(os.path.join(dataset_dir, "home_status_method.jsonl"), "r") as f_home:
                lines_home = f_home.readlines()
            
            home_status = {}
            for line in lines_home:
                data = json.loads(line)
                home_status[data["home_id"]] = {"home_status": data["home_status"], "method": data["method"]}
            
            with open(os.path.join(code_dir, "system.txt"), "r") as f:
                system = f.read()
            
            self.data = []
            for i in range(len(lines)):
                try:
                    case = json.loads(lines[i])
                    if case["home_id"] not in home_status: continue
                    state_str, method_str = chang_json2str(home_status[case["home_id"]]["home_status"], home_status[case["home_id"]]["method"])
                    case_input = "-------------------------------\n" + "Here are the user instructions you need to reply to.\n" + "<User instructions:> \n" + case["input"] + "\n" + "<Machine instructions:>"
                    home_status_case = "<home_state>\n  The following provides the status of all devices in each room of the current household, the adjustable attributes of each device, and the threshold values for adjustable attributes:"+ state_str + "\n" + "</home_state>\n"
                    device_method_case = "<device_method>\n     The following provides the methods to control each device in the current household:"+ method_str + "\n" + "</device_method>\n"
                    full_input = system + home_status_case + device_method_case + case_input
                    self.data.append({"input": full_input, "output": case["output"]})
                except Exception as e:
                    print(f"Error processing line {i}: {e}")
                    continue

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        input_text = [{"role": "system", "content": item["input"]}]
        output_text = item["output"]
        if hasattr(self.tokenizer, "apply_chat_template") and self.tokenizer.chat_template:
            inputs_id = self.tokenizer.apply_chat_template(input_text, add_generation_prompt=True, tokenize=False)
        else:
            inputs_id = item["input"]
        return inputs_id, output_text

# --- 主测试函数 (已修改以支持 DDP) ---
def model_test(model_name, use_rag=False, use_few_shot=False, test_type=None, batch_size=64):
    # --- DDP 初始化检查 ---
    local_rank = int(os.environ.get("LOCAL_RANK", -1))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    ddp_enabled = local_rank != -1

    if ddp_enabled:
        dist.init_process_group(backend="nccl")
        torch.cuda.set_device(local_rank)
        device = torch.device(f"cuda:{local_rank}")
        rank = dist.get_rank()
        if rank == 0:
            print(f"DDP Enabled. World Size: {world_size}")
    else:
        rank = 0
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print("Running in Single Process Mode (No DDP).")

    # 1. 确定路径
    sub_dirs = {
        "llama": "llama3-8b-Instruct",
        "qwen": "Qwen2.5-7B-Instruct",
        "mistral": "Mistral-7B-Instruct-v0.3",
        "gemma": "Gemma-7B-Instruct-v0.3"
    }
    base_model_dir = os.path.join(PROJECT_ROOT, "models", sub_dirs.get(model_name, ""))
    adapter_dir = os.path.join(PROJECT_ROOT, "model_output")
    
    if not os.path.exists(adapter_dir):
        if rank == 0: print(f"Error: Output directory not found at {adapter_dir}")
        return

    # 2. Tokenizer (只打印一次 log)
    tokenizer_path = adapter_dir if os.path.exists(os.path.join(adapter_dir, "tokenizer.json")) else base_model_dir
    if rank == 0: print(f"Loading Tokenizer from: {tokenizer_path}")
    
    tokenizer = None
    try:
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, padding_side='left', trust_remote_code=True)
    except Exception as e:
        if rank == 0: print(f"Standard tokenizer load failed ({e}), trying manual fallback...")
        tokenizer_json = os.path.join(tokenizer_path, "tokenizer.json")
        if os.path.exists(tokenizer_json):
            if "qwen" in model_name.lower():
                tokenizer = Qwen2TokenizerFast(tokenizer_file=tokenizer_json)
            else:
                tokenizer = PreTrainedTokenizerFast(tokenizer_file=tokenizer_json)
            
            token_config_path = os.path.join(tokenizer_path, "tokenizer_config.json")
            if os.path.exists(token_config_path):
                with open(token_config_path, "r") as f:
                    data = json.load(f)
                    if "chat_template" in data: tokenizer.chat_template = data["chat_template"]
            
            base_config = AutoConfig.from_pretrained(base_model_dir, trust_remote_code=True)
            if hasattr(base_config, "pad_token_id") and base_config.pad_token_id is not None:
                tokenizer.pad_token_id = base_config.pad_token_id
            if hasattr(base_config, "eos_token_id"):
                tokenizer.eos_token_id = base_config.eos_token_id
            tokenizer.padding_side = 'left' 
        else:
            raise e

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    # 3. 模型加载 (DDP 模式下加载到特定 GPU)
    if rank == 0: print(f"Loading Base Model from: {base_model_dir}")
    
    # 关键修改：DDP 模式下，不要用 device_map="auto"
    # 我们希望每个进程在自己的 GPU 上加载完整的模型
    load_device_map = None if ddp_enabled else "auto"
    
    model = AutoModelForCausalLM.from_pretrained(
        base_model_dir,
        torch_dtype=torch.bfloat16,
        device_map=load_device_map, 
        trust_remote_code=True
    )

    if ddp_enabled:
        model.to(device) # 显式移动到当前进程的 GPU

    is_lora = os.path.exists(os.path.join(adapter_dir, "adapter_config.json"))
    if is_lora:
        if rank == 0: print(f"Detected LoRA adapter in {adapter_dir}. Loading adapter...")
        try:
            model = PeftModel.from_pretrained(model, adapter_dir)
            if ddp_enabled:
                model.to(device) # 确保 adapter 也在正确的设备上
            if rank == 0: print("Successfully loaded LoRA adapter!")
        except Exception as e:
            if rank == 0: print(f"Failed to load LoRA adapter: {e}")
            return

    # 4. 数据集与 DistributedSampler
    if rank == 0: print("Loading test dataset...")
    test_dataset = no_few_shot_home_assistant_dataset(tokenizer, use_rag=use_rag)
    if rank == 0: print(f"Test dataset size: {len(test_dataset)}")
    
    sampler = None
    if ddp_enabled:
        # 关键：使用 DistributedSampler 切分数据
        sampler = DistributedSampler(test_dataset, shuffle=False) # shuffle=False 保持顺序（虽然多卡合并时还是会乱，需要后处理排序或不排序直接合并）
    
    # 增大 Batch Size (A800 80G 很大，可以开大一点)
    # 使用传入的 batch_size，默认为 64
    test_loader = DataLoader(test_dataset, batch_size=batch_size, sampler=sampler, shuffle=False)
    
    res = []
    
    # 只有 rank 0 显示进度条，避免刷屏
    iterator = tqdm(test_loader, disable=(rank != 0))
    
    start_time = time.time()
    
    for inputs_str, output_text in iterator:
        # inputs_str 已经在 device 上了吗？不，需要 tokenizing
        inputs = tokenizer(list(inputs_str), return_tensors="pt", padding=True).to(device)
        
        with torch.no_grad():
            logits = model.generate(
                **inputs,
                max_new_tokens=512,
                do_sample=True,
                temperature=0.1,
                top_p=0.9,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id
            )
        
        response = logits[:, inputs['input_ids'].shape[1]:]
        generated_texts = tokenizer.batch_decode(response, skip_special_tokens=True)

        for i in range(len(generated_texts)):
            res.append({"generated": generated_texts[i], "expected": output_text[i]})
            
    end_time = time.time()
    if rank == 0: print(f"Total Inference Time: {end_time - start_time:.2f}s")
    
    # 5. 保存结果 (每个 Rank 保存自己的部分，然后合并)
    output_dir = os.path.join(PROJECT_ROOT, "output")
    os.makedirs(output_dir, exist_ok=True)
    
    # 保存分片文件
    part_file = os.path.join(output_dir, f"{model_name}_{test_type}_part_{rank}.json")
    with open(part_file, "w") as f:
        f.write(json.dumps(res, indent=4, ensure_ascii=False))
    
    # 等待所有进程写完
    if ddp_enabled:
        dist.barrier()
    
    # Rank 0 负责合并
    if rank == 0:
        print("Merging results from all ranks...")
        final_res = []
        # 查找所有 part 文件
        pattern = os.path.join(output_dir, f"{model_name}_{test_type}_part_*.json")
        for file_path in glob.glob(pattern):
            try:
                with open(file_path, "r") as f:
                    part_data = json.load(f)
                    final_res.extend(part_data)
                os.remove(file_path) # 合并后删除分片
            except Exception as e:
                print(f"Error reading {file_path}: {e}")
        
        final_file = os.path.join(output_dir, f"{model_name}_{test_type}_test_result.json")
        with open(final_file, "w") as f:
            f.write(json.dumps(final_res, indent=4, ensure_ascii=False))
        print(f"Final merged results saved to: {final_file} (Total records: {len(final_res)})")

    if ddp_enabled:
        dist.destroy_process_group()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Model Testing Script for HomeBench.")
    parser.add_argument("--model_name", type=str, default="qwen", 
                        choices=["llama", "qwen", "mistral", "gemma"], 
                        help="Name of the model to test.")
    parser.add_argument("--use_rag", action="store_true", help="Use RAG.")
    parser.add_argument("--use_few_shot", action="store_true", help="Use Few-Shot.")
    parser.add_argument("--test_type", type=str, default="normal", help="Type of test.")
    parser.add_argument("--cuda_devices", type=str, default="0,1", help="CUDA devices.")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size per device for testing.")

    args = parser.parse_args()
    
    # 关键：如果是 torchrun 启动，不要手动覆盖 CUDA_VISIBLE_DEVICES，
    # 除非你确定你在做什么。DDP 环境下通常由外部控制或默认可见所有。
    # 这里我们做一个判断：如果没有 LOCAL_RANK 才设置
    if os.environ.get("LOCAL_RANK") is None:
        os.environ['CUDA_VISIBLE_DEVICES'] = args.cuda_devices
        print(f"Using CUDA devices: {args.cuda_devices}")
    
    model_test(args.model_name, use_rag=args.use_rag, use_few_shot=args.use_few_shot, test_type=args.test_type, batch_size=args.batch_size)