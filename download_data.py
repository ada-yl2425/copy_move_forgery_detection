# download_data.py
import os
import config
import zipfile

# 确保 Kaggle API 密钥 (kaggle.json) 已放置

print(f"将要下载数据到: {config.DATA_DIR}")
config.DATA_DIR.mkdir(exist_ok=True)

# 1. 下载
# (你可能需要先接受 Kaggle 竞赛规则)
os.system(f"kaggle competitions download -c recodai-luc-scientific-image-forgery-detection -p {config.DATA_DIR}")

# 2. 解压
zip_path = config.DATA_DIR / "recodai-luc-scientific-image-forgery-detection.zip"
extract_path = config.FORGERY_DATA_ROOT 

if zip_path.exists():
    print(f"正在解压 {zip_path} 到 {extract_path}...")
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_path)
    print("解压完成。")
    
    os.remove(zip_path)
else:
    print(f"错误: 未找到 {zip_path}。请确保 Kaggle API 配置正确。")

print("数据准备脚本完成。")
