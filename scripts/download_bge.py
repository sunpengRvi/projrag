import os
from modelscope import snapshot_download

# 我们把模型统一下载到项目根目录的 models 文件夹下
local_model_dir = os.path.join(os.getcwd(), "models")

print("[*] 开始从 ModelScope (魔搭) 下载 BGE-m3 模型...")
print(f"[*] 目标存储路径: {local_model_dir}")

# BAAI/bge-m3 是智源研究院在魔搭上的官方仓库
# snapshot_download 会自动处理断点续传
model_path = snapshot_download(
    'BAAI/bge-m3', 
    cache_dir=local_model_dir
)

print(f"[+] 下载完成！模型已保存在: {model_path}")
print("[+] 请将这个路径复制到 ingest_data.py 中。")