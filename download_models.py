import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
from huggingface_hub import snapshot_download

# 定义模型列表，包含 Hugging Face 的 repo_id 和在本地 models 文件夹下的子目录名
MODELS_TO_DOWNLOAD = [
    {"repo_id": "meta-llama/Meta-Llama-3-8B-Instruct", "sub_dir": "llama3-8b-Instruct"},
    {"repo_id": "Qwen/Qwen2.5-7B-Instruct", "sub_dir": "Qwen2.5-7B-Instruct"},
    {"repo_id": "mistralai/Mistral-7B-Instruct-v0.3", "sub_dir": "Mistral-7B-Instruct-v0.3"},
    {"repo_id": "google/gemma-7b-it", "sub_dir": "Gemma-7B-Instruct-v0.3"},
]

def download_all_models():
    # 获取当前脚本所在目录，即 HomeBenchReproduction 目录
    script_dir = os.path.dirname(os.path.abspath(__file__))
    models_dir = os.path.join(script_dir, "models")

    # 如果 models 文件夹不存在，则创建
    os.makedirs(models_dir, exist_ok=True)
    print(f"确保 models 目录存在: {models_dir}")

    for model_info in MODELS_TO_DOWNLOAD:
        repo_id = model_info["repo_id"]
        sub_dir = model_info["sub_dir"]
        local_model_path = os.path.join(models_dir, sub_dir)

        print(f"开始下载模型: {repo_id} 到 {local_model_path}")
        try:
            snapshot_download(
                repo_id=repo_id,
                local_dir=local_model_path,
                local_dir_use_symlinks=False, # 建议设置为 False，以确保所有文件都被复制而不是链接
                resume_download=True # 允许断点续传
            )
            print(f"成功下载模型: {repo_id}")
        except Exception as e:
            print(f"下载模型 {repo_id} 失败: {e}")
            print("请确保您已登录 Hugging Face 并设置了 HF_TOKEN 环境变量，尤其是对于 Llama 模型。")

if __name__ == "__main__":
    download_all_models()
