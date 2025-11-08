# train.py
import torch
import config  # 导入你的配置文件
import json
import os
from pathlib import Path

# --- 从 src 模块导入所有必要的类和函数 ---
from src.data import load_and_rebuild_data
from src.model import BusterNetCMFD
from src.loss import CombinedLoss  # 确保 CombinedLoss 在 src/loss.py 中
from src.trainer import CopyMoveTrainer
from src.post_process import PostProcessor, predict_with_postprocessing

def main():
    """主训练和预测程序"""

    # --- 1. 设置 ---
    if torch.cuda.is_available():
        device = torch.device(config.DEVICE)
        print(f"使用设备: NVIDIA CUDA (GPU) - {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device("cpu")
        print("警告: 未检测到 CUDA。使用设备: CPU")

    print(f"--- 路径配置 ---")
    print(f"JSON 目录: {config.DATA_SPLITS_DIR}")
    print(f"数据根目录: {config.DATA_DIR}") # 这是 forgery_data 的父目录
    print(f"模型保存路径: {config.BEST_MODEL_PATH}")
    print(f"--------------------")

    # --- 2. 加载数据 ---
    # (确保 load_and_rebuild_data 从 config.DATA_DIR 正确拼接路径)
    train_loader, val_loader, data_stats = load_and_rebuild_data(
        json_dir=config.DATA_SPLITS_DIR,
        data_root_on_drive=config.DATA_DIR 
    )

    if train_loader is None:
        raise RuntimeError("数据加载失败，请检查你的路径配置和JSON文件。")

    # --- 3. 初始化模型、损失和训练器 ---
    model = BusterNetCMFD(feature_dim=256)
    print("已实例化 BusterNetCMFD (U-Net 架构)。")

    class_weights_list = data_stats.get('class_weights', None)
    class_weights_tensor = None
    if class_weights_list:
        class_weights_tensor = torch.tensor(class_weights_list)

    trainer = CopyMoveTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        class_weights=class_weights_tensor,
        lr=config.LEARNING_RATE,
        patience=config.PATIENCE,
        T_0=10, # 你可以将其移到 config.py
        best_model_save_path=config.BEST_MODEL_PATH # 传入配置好的路径
    )

    # --- 4. 开始训练 ---
    print("\n---------------------------------")
    print("开始训练...")
    
    for epoch in range(1, config.NUM_EPOCHS + 1):
        print(f'--- Epoch {epoch}/{config.NUM_EPOCHS} ---')
        train_loss, train_components = trainer.train_epoch(epoch)
        val_loss, val_iou = trainer.validate(epoch)
        print(f'  Epoch {epoch} 摘要:')
        print(f'    训练损失: {train_loss:.4f} (BCE: {train_components["bce"]:.4f}, Dice: {train_components["dice"]:.4f}, Focal: {train_components["focal"]:.4f})')
        print(f'    验证损失: {val_loss:.4f}, 验证 Micro IoU: {val_iou:.4f}')
        print("---------------------------------")
        if trainer.should_stop():
            print(f"早停在 epoch {epoch}")
            break
    print("训练完成!")

    # --- 5. 训练后运行预测 ---
    print("\n---------------------------------")
    print("在验证集上运行后处理和RLE编码...")
    post_processor = PostProcessor(min_area=100) # 你也可以在 config 中配置

    if os.path.exists(config.BEST_MODEL_PATH):
        print(f"加载最佳模型从: {config.BEST_MODEL_PATH}")
        # (确保 load_model 函数在 src/utils.py 中)
        # 这里我们直接加载，因为 trainer 已经保存了它
        model_for_pred = BusterNetCMFD(feature_dim=256)
        model_for_pred.load_state_dict(torch.load(config.BEST_MODEL_PATH, map_location=device))
        model_for_pred.to(device)
        model_for_pred.eval()
    else:
        print(f"!! 错误: 未找到 '{config.BEST_MODEL_PATH}'。")
        model_for_pred = None

    if model_for_pred:
        val_predictions = predict_with_postprocessing(
            model_for_pred, val_loader, device, post_processor
        )
        print(f"已生成 {len(val_predictions)} 条预测。")

        # --- 6. 保存预测结果 ---
        print(f"正在保存预测到: {config.PREDICTIONS_JSON_PATH}")
        try:
            with open(config.PREDICTIONS_JSON_PATH, 'w') as f:
                json.dump(val_predictions, f, indent=4)
            print(f"预测结果已成功保存。")
        except Exception as e:
            print(f"保存 JSON 预测时出错: {e}")
    else:
        print("跳过预测步骤，因为模型未加载。")

    print("训练脚本执行完毕。")

if __name__ == '__main__':
    main()
