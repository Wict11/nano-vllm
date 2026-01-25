import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
from huggingface_hub import snapshot_download

# 换成这个官方公开的 MoE 模型
model_id = "Qwen/Qwen1.5-MoE-A2.7B" 
local_dir = "/mnt/workspace/nano_vllm/nano-vllm/Qwen/Qwen-MoE-Test"

print(f"开始下载公开 MoE 模型: {model_id} ...")

try:
    snapshot_download(
        repo_id=model_id,
        local_dir=local_dir,
        local_dir_use_symlinks=False,
        resume_download=True,
        token=False # 明确表示不需要 token
    )
    print(f"\n下载成功！路径: {local_dir}")
except Exception as e:
    print(f"\n下载失败: {e}")