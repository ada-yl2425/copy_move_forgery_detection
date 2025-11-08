# evaluate.py
import torch
import config # 导入你的配置文件
import time

# --- 从 src 模块导入所有必要的类和函数 ---
from src.data import load_validation_data
from src.utils import load_model 
from src.visualization import visualize_predictions
from src.post_process import PostProcessor, get_evaluation_scores
from src.model import BusterNetCMFD # load_model 需要这个定义

def main():
    """主评估和可视化程序"""
    
    # --- 1. 设置 ---
    if torch.cuda.is_available():
        device = torch.device(config.DEVICE)
        print(f"使用设备: NVIDIA CUDA (GPU) - {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device("cpu")
        print("警告: 未检测到 CUDA。使用设备: CPU")

    # --- 2. 加载数据 ---
    print("--- 正在加载验证数据 ---")
    # (确保 load_validation_data 已从 evaluation.py 添加到 src/data.py)
    val_loader, norm_mean, norm_std = load_validation_data(
        json_dir=config.DATA_SPLITS_DIR,
        data_root_on_drive=config.DATA_DIR,
        batch_size=config.BATCH_SIZE,
        num_workers=config.NUM_WORKERS
    )
    if val_loader is None:
        print("!! 数据加载失败，退出。")
        return

    # --- 3. 加载模型 ---
    print(f"--- 正在加载模型: {config.BEST_MODEL_PATH} ---")
    # (确保 load_model 已从 evaluation.py 添加到 src/utils.py)
    model = load_model(config.BEST_MODEL_PATH, device)
    post_processor = PostProcessor(min_area=100) # 也可以在 config 中配置

    if model is None:
        print("!! 模型加载失败，退出。")
        return
        
    # --- 4. 运行可视化 ---
    print("\n--- 开始可视化 (10 个样本) ---")
    # (确保 visualize_predictions 在 src/visualization.py 中)
    visualize_predictions(
        model=model,
        dataloader=val_loader,
        device=device,
        post_processor=post_processor,
        norm_mean=norm_mean,
        norm_std=norm_std,
        num_samples=10
    )

    # --- 5. 运行完整评估 ---
    print("\n--- 开始在整个验证集上进行评分 ---")
    start_time = time.time()
    # (确保 get_evaluation_scores 在 src/post_process.py 中)
    get_evaluation_scores(
        model=model,
        dataloader=val_loader,
        device=device,
        post_processor=post_processor
    )
    end_time = time.time()
    print(f"评分总耗时: {end_time - start_time:.2f} 秒")
    print("评估脚本执行完毕。")


if __name__ == '__main__':
    main()
