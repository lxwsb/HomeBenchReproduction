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
from peft import PeftModel

# --- 路径设置 ---
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
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

# --- Dataset 类 1: Zero-shot (No Few-Shot) ---
class no_few_shot_home_assistant_dataset(Dataset):
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        dataset_dir = os.path.join(PROJECT_ROOT, "dataset")
        code_dir = os.path.join(PROJECT_ROOT, "code")
        
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
                
                # Zero-shot Prompt 结构
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

# --- Dataset 类 2: Few-shot (ICL) ---
class home_assistant_dataset(Dataset):
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        dataset_dir = os.path.join(PROJECT_ROOT, "dataset")
        code_dir = os.path.join(PROJECT_ROOT, "code")
        
        with open(os.path.join(dataset_dir, "test_data.jsonl"), "r") as f:
            lines = f.readlines()
        with open(os.path.join(dataset_dir, "home_status_method.jsonl"), "r") as f_home:
            lines_home = f_home.readlines()
        
        home_status = {}
        for line in lines_home:
            data = json.loads(line)
            home_status[data["home_id"]] = {"home_status": data["home_status"], "method": data["method"]}
        
        # 加载示例文件 (ICL 核心)
        ex_path1 = os.path.join(code_dir, "example1.txt")
        ex_path2 = os.path.join(code_dir, "example.txt")
        if os.path.exists(ex_path1):
            with open(ex_path1, "r") as f: examples = f.read()
        elif os.path.exists(ex_path2):
            with open(ex_path2, "r") as f: examples = f.read()
        else:
            print("Warning: No example file found for Few-shot mode!")
            examples = ""

        with open(os.path.join(code_dir, "system.txt"), "r") as f:
            system = f.read()
        
        self.data = []
        for i in range(len(lines)):
            try:
                case = json.loads(lines[i])
                if case["home_id"] not in home_status: continue
                state_str, method_str = chang_json2str(home_status[case["home_id"]]["home_status"], home_status[case["home_id"]]["method"])
                
                # Few-shot Prompt 结构
                case_input = "-------------------------------\n" + "Here are the user instructions you need to reply to.\n" + "<User instructions:> \n" + case["input"] + "\n" + "<Machine instructions:>"
                home_status_case = "<home_state>\n  The following provides the status of all devices in each room of the current household, the adjustable attributes of each device, and the threshold values for adjustable attributes:"+ state_str + "\n" + "</home_state>\n"
                device_method_case = "<device_method>\n     The following provides the methods to control each device in the current household:"+ method_str + "\n" + "</device_method>\n"
                
                full_input = system + home_status_case + device_method_case + examples + case_input
                self.data.append({"input": full_input, "output": case["output"]})
            except Exception as e:
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

# --- Dataset 类 3: RAG Mode ---
class rag_home_assistant_dataset(Dataset):
    def __init__(self, tokenizer, model_name_for_rag):
        self.tokenizer = tokenizer
        dataset_dir = os.path.join(PROJECT_ROOT, "dataset")
        
        filename_candidates = [
            f"{model_name_for_rag}_rag_test_data.json", 
            "rag_test_data.json" 
        ]
        
        rag_path = None
        for fn in filename_candidates:
            p = os.path.join(dataset_dir, fn)
            if os.path.exists(p):
                rag_path = p
                break
        
        if not rag_path:
            candidates = glob.glob(os.path.join(dataset_dir, "*rag_test_data.json"))
            if candidates:
                rag_path = candidates[0]
        
        if not rag_path:
            raise FileNotFoundError(f"Could not find RAG dataset file in {dataset_dir}. Please run rag_dataset_generation.py first.")
            
        print(f"Loading RAG dataset from: {rag_path}")
        with open(rag_path, "r") as f:
            self.data = json.load(f)

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

# --- 主测试函数 ---
def model_test(model_name, use_rag=False, use_few_shot=False, use_finetuned=False, test_type=None, batch_size=64):
    local_rank = int(os.environ.get("LOCAL_RANK", -1))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    ddp_enabled = local_rank != -1

    if ddp_enabled:
        dist.init_process_group(backend="nccl")
        torch.cuda.set_device(local_rank)
        device = torch.device(f"cuda:{local_rank}")
        rank = dist.get_rank()
        if rank == 0: print(f"DDP Enabled. World Size: {world_size}")
    else:
        rank = 0
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print("Running in Single Process Mode (No DDP).")

    sub_dirs = {
        "llama": "llama3-8b-Instruct",
        "qwen": "Qwen2.5-7B-Instruct",
        "mistral": "Mistral-7B-Instruct-v0.3",
        "gemma": "Gemma-7B-Instruct-v0.3"
    }
    
    base_model_path_name = sub_dirs.get(model_name, model_name)
    base_model_dir = os.path.join(PROJECT_ROOT, "models", base_model_path_name)
    adapter_dir = os.path.join(PROJECT_ROOT, "model_output")
    
    if rank == 0: 
        print(f"Test Config: RAG={use_rag}, Few-Shot={use_few_shot}, SFT_Adapter={use_finetuned}")
        print(f"Loading Base Model from: {base_model_dir}")

    # Tokenizer 加载 (优先加载 SFT 后的 tokenizer 以防有新增 token，否则加载基座的)
    tokenizer_source = adapter_dir if use_finetuned and os.path.exists(os.path.join(adapter_dir, "tokenizer.json")) else base_model_dir
    try:
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_source, padding_side='left', trust_remote_code=True)
    except:
        tokenizer = Qwen2TokenizerFast(tokenizer_file=os.path.join(tokenizer_source, "tokenizer.json"))
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    # Model 加载
    load_device_map = None if ddp_enabled else "auto"
    model = AutoModelForCausalLM.from_pretrained(
        base_model_dir,
        torch_dtype=torch.bfloat16,
        device_map=load_device_map, 
        trust_remote_code=True
    )
    if ddp_enabled: model.to(device)

    # --- 关键修改：显式控制 LoRA 加载 ---
    if use_finetuned:
        if os.path.exists(os.path.join(adapter_dir, "adapter_config.json")):
            if rank == 0: print(f"Loading SFT Adapter from {adapter_dir}...")
            model = PeftModel.from_pretrained(model, adapter_dir)
            if ddp_enabled: model.to(device)
        else:
            if rank == 0: print(f"Warning: --use_finetuned is set but no adapter found in {adapter_dir}. Running with Base Model.")
    else:
        if rank == 0: print("Running with Base Model (Zero-shot/Few-shot Baseline Mode). Adapter NOT loaded.")

    # --- 数据集选择逻辑 ---
    if rank == 0: print("Loading test dataset...")
    
    if use_rag:
        # RAG 模式
        test_dataset = rag_home_assistant_dataset(tokenizer, base_model_path_name)
    elif use_few_shot:
        # ICL 模式
        test_dataset = home_assistant_dataset(tokenizer)
    else:
        # Zero-shot / SFT 模式
        test_dataset = no_few_shot_home_assistant_dataset(tokenizer)
        
    if rank == 0: print(f"Dataset size: {len(test_dataset)}")
    
    sampler = DistributedSampler(test_dataset, shuffle=False) if ddp_enabled else None
    test_loader = DataLoader(test_dataset, batch_size=batch_size, sampler=sampler, shuffle=False)
    
    # --- 推理循环 ---
    res = []
    iterator = tqdm(test_loader, disable=(rank != 0))
    start_time = time.time()
    
    for inputs_str, output_text in iterator:
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
            
    if rank == 0: print(f"Inference Time: {time.time() - start_time:.2f}s")
    
    # --- 保存结果 ---
    output_dir = os.path.join(PROJECT_ROOT, "output")
    os.makedirs(output_dir, exist_ok=True)
    
    # 构造输出文件名
    mode_parts = []
    if use_finetuned: mode_parts.append("sft")
    if use_rag: mode_parts.append("rag")
    elif use_few_shot: mode_parts.append("few_shot")
    else: mode_parts.append("zero_shot")
    
    mode_suffix = "_".join(mode_parts)
    if test_type and test_type != "normal": mode_suffix += f"_{test_type}"
    
    part_file = os.path.join(output_dir, f"{model_name}_{mode_suffix}_part_{rank}.json")
    with open(part_file, "w") as f:
        f.write(json.dumps(res, indent=4, ensure_ascii=False))
    
    if ddp_enabled: dist.barrier()
    
    if rank == 0:
        print("Merging results...")
        final_res = []
        pattern = os.path.join(output_dir, f"{model_name}_{mode_suffix}_part_*.json")
        for file_path in glob.glob(pattern):
            try:
                with open(file_path, "r") as f:
                    final_res.extend(json.load(f))
                os.remove(file_path)
            except: pass
        
        final_file = os.path.join(output_dir, f"{model_name}_{mode_suffix}_test_result.json")
        with open(final_file, "w") as f:
            f.write(json.dumps(final_res, indent=4, ensure_ascii=False))
        print(f"Saved to: {final_file}")

    if ddp_enabled: dist.destroy_process_group()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="qwen", choices=["llama", "qwen", "mistral", "gemma"])
    parser.add_argument("--use_rag", action="store_true")
    parser.add_argument("--use_few_shot", action="store_true")
    parser.add_argument("--use_finetuned", action="store_true", help="Load fine-tuned LoRA adapter.") # 新增参数
    parser.add_argument("--test_type", type=str, default="normal")
    parser.add_argument("--cuda_devices", type=str, default="0,1")
    parser.add_argument("--batch_size", type=int, default=64)
    args = parser.parse_args()
    
    if os.environ.get("LOCAL_RANK") is None:
        os.environ['CUDA_VISIBLE_DEVICES'] = args.cuda_devices
    
    model_test(args.model_name, args.use_rag, args.use_few_shot, args.use_finetuned, args.test_type, args.batch_size)