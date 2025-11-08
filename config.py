# config.py
from pathlib import Path

# --- 1. 基础路径 ---
# (Path.cwd() 会获取你运行 train.py 的当前目录)
ROOT_DIR = Path.cwd() 

# --- 2. Git 跟踪的目录 ---
# (你存放 JSON 元数据的地方)
ASSETS_DIR = ROOT_DIR / "assets"
DATA_SPLITS_DIR = ASSETS_DIR / "data_splits"

# --- 3. Git 忽略的目录 ---
# (Kaggle 数据下载到这里)
DATA_DIR = ROOT_DIR / "data"
FORGERY_DATA_ROOT = DATA_DIR / "forgery_data" # 指向 'forgery_data' 文件夹

# (模型和预测保存到这里)
OUTPUT_DIR = ROOT_DIR / "outputs"
MODEL_DIR = OUTPUT_DIR / "trained_models"
PREDICTION_DIR = OUTPUT_DIR / "predictions"

# 确保目录存在
MODEL_DIR.mkdir(parents=True, exist_ok=True)
PREDICTION_DIR.mkdir(parents=True, exist_ok=True)

# --- 4. 最终文件路径 ---
# (你的脚本将使用这些变量)
BEST_MODEL_PATH = MODEL_DIR / "best_model_bay.pth"
PREDICTIONS_JSON_PATH = PREDICTION_DIR / "predictions.json"

# --- 5. 训练超参数 ---
DEVICE = "cuda"
IMG_SIZE = (512, 512)
BATCH_SIZE = 8
NUM_WORKERS = 2
LEARNING_RATE = 1e-4
PATIENCE = 15
NUM_EPOCHS = 50
