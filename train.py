"""
NBV框架演示脚本

展示如何使用目标驱动的NBV策略学习框架进行训练和评估。
"""

import torch
import torch.multiprocessing as mp
import time
import os
import argparse
import random
import numpy as np
from typing import Dict, Any


def set_random_seed(seed: int = 42):
    """设置所有随机种子以确保实验可重现性"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # 设置PyTorch的确定性行为
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    print(f"Random seed set to {seed} for reproducibility")


# 导入NBV框架组件
from nbv_framework import VGGTWrapper,BaseNBVPolicy, BasicNBVPolicy, DifferentiableRenderer, NBVTrainer
from nbv_framework.training.losses import ReconstructionLoss
from nbv_framework.datasets import SyntheticDataset, MixedDataset
from nbv_framework.datasets.data_loaders import create_train_loader, create_val_loader
from nbv_framework.utils.data_utils import create_synthetic_training_data
from nbv_framework.utils.visualization import visualize_reconstruction, plot_training_curves
from nbv_framework.utils.evaluation import evaluate_nbv_policy, compare_with_baselines


def setup_config() -> Dict[str, Any]:
    """设置配置参数"""
    experiment_name = "dataset-house3k"
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    config = {
        # 模型配置
        "scene_feature_dim": 2048,
        "policy_hidden_dim": 256,
        "policy_num_layers": 3,
        # "policy_output_mode": "cartesian",
        "policy_output_mode": "position_only",
        
        # 训练配置
        "learning_rate": 1e-3,
        "batch_size": 1,  # 根据GPU内存调整
        "num_epochs": 1000,
        "num_samples": 20000,
        "weight_decay": 1e-5,
        
        # 数据配置
        "synthetic_data_root": "./models/synthetic_data",
        "num_initial_views": 3,
        "image_size": 224,
        "up_axis": "Y",  # 数据集模型默认上方向 ('Y' 或 'Z')
        "max_meshes": 2,  # 限制加载的mesh数量，用于控制训练规模
        
        # 设备配置
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        
        # 断点续训配置
        "resume_checkpoint": None,  # 指定要恢复的检查点路径
        "auto_resume": False,       # 是否自动从最新检查点恢复
    }
    
    # 在config定义完成后设置路径相关配置，避免循环引用
    config["save_dir"] = f"./checkpoints/{experiment_name}_bs-{config['batch_size']}_initv-{config['num_initial_views']}_pom-{config['policy_output_mode']}_{timestamp}"
    config["log_dir"] = f"runs/{experiment_name}_bs-{config['batch_size']}_initv-{config['num_initial_views']}_pom-{config['policy_output_mode']}_{timestamp}"
    
    return config


def create_synthetic_data(config: Dict[str, Any]):
    """创建合成训练数据"""
    print("Creating synthetic training data...")
    
    create_synthetic_training_data(
        output_dir=config["synthetic_data_root"],
        num_objects=20,  # 小规模演示
        num_views_per_object=15,
        image_size=config["image_size"],
    )
    
    print("Synthetic data created successfully!")


def setup_models(config: Dict[str, Any]):
    """设置模型组件"""
    device = config["device"]
    
    print(f"Setting up models on device: {device}")
    
    # 1. VGGT基础模型（冻结）
    print("Loading VGGT wrapper...")
    vggt_wrapper = VGGTWrapper(
        model_name="facebook/VGGT-1B",
        device=device
    )
    
    # 2. NBV策略网络（可训练）
    print("Creating NBV policy network...")
    policy_network = BasicNBVPolicy(
        scene_feature_dim=config["scene_feature_dim"],
        hidden_dim=config["policy_hidden_dim"],
        num_layers=config["policy_num_layers"],
        output_mode=config["policy_output_mode"]
    ).to(device)
    
    # 3. 可微分渲染器
    print("Setting up differentiable renderer...")
    renderer = DifferentiableRenderer(
        image_size=config["image_size"],
        device=device,
        quality="high",
        downsample_factor=2
    )
    
    # 4. 损失函数
    loss_fn = ReconstructionLoss(
        renderer=renderer
    )
    
    print("Models setup completed!")
    
    return vggt_wrapper, policy_network, renderer, loss_fn


def setup_data_loaders(config: Dict[str, Any]):
    """设置数据加载器"""
    print("Setting up data loaders...")
    
    # 可以选择使用不同的数据集类型
    # 选项1: 合成数据集
    # dataset_train_configs = [
    #     {
    #         "name": "synthetic_data",
    #         "type": "synthetic",
    #         "data_root": config["data_root"], 
    #         "num_initial_views": config["num_initial_views"],
    #         "image_size": config["image_size"],
    #         "split": "train"
    #     }
    # ]
    
    # 选项2: House3K数据集
    dataset_train_configs = [
        {
            "name": "house3k_data",
            "type": "house3k",
            "data_root": "/mnt/sdb/chenmohan/VGGT-NBV/models/House3K_obj",
            "num_initial_views": config["num_initial_views"],
            "image_size": config["image_size"],
            "normalize_method": "quantile",
            "num_samples": config["num_samples"],
            "split": "train",
            "max_meshes": config.get("max_meshes", 100),  # 限制总mesh数量
            "use_cache": True,
            "up_axis": config.get("up_axis", "Y")  # 数据集模型默认上方向
        }
    ]

    dataset_val_configs = [
        {
            "name": "house3k_data",
            "type": "house3k",
            "data_root": "/mnt/sdb/chenmohan/VGGT-NBV/models/House3K_obj",
            "num_initial_views": config["num_initial_views"],
            "image_size": config["image_size"],
            "normalize_method": "quantile",
            "num_samples": config["num_samples"],
            "split": "val",
            "max_meshes": config.get("max_meshes", 100),  # 验证集使用更少的mesh
            "use_cache": True,
            "up_axis": config.get("up_axis", "Y")  # 数据集模型默认上方向
        }
    ]

    # 训练数据集
    train_dataset = MixedDataset(
        dataset_configs=dataset_train_configs,
        seed=42
    )
    
    # 验证数据集
    val_dataset = MixedDataset(
        dataset_configs=dataset_val_configs,
        seed=42
    )
    
    # 数据加载器
    train_loader = create_train_loader(
        train_dataset,
        batch_size=config["batch_size"],
        num_workers=1,
    )
    
    val_loader = create_val_loader(
        val_dataset,
        batch_size=config["batch_size"],
        num_workers=1,
    )
    
    print(f"Data loaders created - Train: {len(train_dataset)}, Val: {len(val_dataset)}")
    
    return train_loader, val_loader


def find_latest_checkpoint(save_dir: str) -> str:
    """查找最新的检查点文件"""
    if not os.path.exists(save_dir):
        return None
    
    checkpoint_files = []
    for file in os.listdir(save_dir):
        if file.startswith("checkpoint_epoch_") and file.endswith(".pth"):
            # 提取epoch数字
            try:
                epoch_num = int(file.replace("checkpoint_epoch_", "").replace(".pth", ""))
                checkpoint_files.append((epoch_num, os.path.join(save_dir, file)))
            except ValueError:
                continue
    
    if checkpoint_files:
        # 返回最新的检查点
        latest_checkpoint = max(checkpoint_files, key=lambda x: x[0])
        return latest_checkpoint[1]
    
    return None


def train_nbv_policy(config: Dict[str, Any],
                    vggt_wrapper: VGGTWrapper,
                    policy_network: BaseNBVPolicy,
                    renderer: DifferentiableRenderer,
                    loss_fn: ReconstructionLoss,
                    train_loader,
                    val_loader):
    """训练NBV策略"""
    print("Starting NBV policy training...")
    
    # 创建训练器
    trainer = NBVTrainer(
        vggt_wrapper=vggt_wrapper,
        policy_network=policy_network,
        renderer=renderer,
        loss_fn=loss_fn,
        num_epochs=config["num_epochs"],
        learning_rate=config["learning_rate"],
        weight_decay=config["weight_decay"],
        device=config["device"],
        log_dir=config["log_dir"]
    )
    
    # 断点续训逻辑
    resume_checkpoint_path = None
    
    # 1. 检查是否指定了特定的检查点
    if config.get("resume_checkpoint") and os.path.exists(config["resume_checkpoint"]):
        resume_checkpoint_path = config["resume_checkpoint"]
        print(f"Resuming from specified checkpoint: {resume_checkpoint_path}")
    
    # 2. 如果启用自动恢复，查找最新检查点
    elif config.get("auto_resume", True):
        latest_checkpoint = find_latest_checkpoint(config["save_dir"])
        if latest_checkpoint:
            resume_checkpoint_path = latest_checkpoint
            print(f"Auto-resuming from latest checkpoint: {resume_checkpoint_path}")
    
    # 3. 加载检查点
    if resume_checkpoint_path:
        try:
            trainer.load_checkpoint(resume_checkpoint_path)
            print(f"Successfully resumed from epoch {trainer.current_epoch}")
        except Exception as e:
            print(f"Failed to load checkpoint: {e}")
            print("Starting training from scratch...")
    else:
        print("No checkpoint found, starting training from scratch...")
    
    # 开始训练
    trainer.train(
        train_loader=train_loader,
        val_loader=val_loader,
        save_dir=config["save_dir"]
    )
    
    print("Training completed!")
    
    return trainer

def run_evaluation(config: Dict[str, Any],
                  vggt_wrapper: VGGTWrapper,
                  policy_network: BaseNBVPolicy,
                  renderer: DifferentiableRenderer):
    """运行评估"""
    print("Running evaluation...")
    
    # 创建测试数据集
    test_dataset = SyntheticDataset(
        data_root=config["synthetic_data_root"],
        num_initial_views=config["num_initial_views"],
        image_size=config["image_size"],
        split="val"  # 使用验证集作为测试集
    )
    
    # 准备测试数据
    test_data = []
    for i in range(min(5, len(test_dataset))):  # 只测试前5个样本
        test_data.append(test_dataset[i])
    
    # 评估策略性能
    evaluation_results = evaluate_nbv_policy(
        policy_network=policy_network,
        vggt_wrapper=vggt_wrapper,
        renderer=renderer,
        test_data=test_data,
        max_views=8,
        device=config["device"]
    )
    
    print("Evaluation Results:")
    for metric, value in evaluation_results.items():
        print(f"  {metric}: {value:.6f}")
    
    # 与基线方法比较
    print("Comparing with baseline methods...")
    comparison_results = compare_with_baselines(
        policy_network=policy_network,
        vggt_wrapper=vggt_wrapper,
        renderer=renderer,
        test_data=test_data[:3],  # 减少样本数量以节省时间
        device=config["device"]
    )
    
    print("Comparison Results:")
    for method, results in comparison_results.items():
        print(f"\n{method.upper()}:")
        for metric, value in results.items():
            print(f"  {metric}: {value:.6f}")
    
    print("Evaluation completed!")


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="NBV Framework Demo")
    parser.add_argument("--mode", type=str, choices=["train", "eval", "all"],
                       default="train", help="Running mode")
    parser.add_argument("--create_data", action="store_true", 
                       help="Create synthetic training data")
    parser.add_argument("--resume", type=str, default=None,
                       help="Resume from specified checkpoint path")
    parser.add_argument("--no_auto_resume", action="store_true",
                       help="Disable automatic resume from latest checkpoint")
    parser.add_argument("--seed", type=int, default=42,
                       help="Random seed for reproducibility")
    
    args = parser.parse_args()
    
    # 设置随机种子（在所有其他操作之前）
    set_random_seed(args.seed)
    
    # 设置配置
    config = setup_config()
    
    # 更新断点续训配置
    if args.resume:
        config["resume_checkpoint"] = args.resume
        print(f"Resume checkpoint specified: {args.resume}")
    
    if args.no_auto_resume:
        config["auto_resume"] = False
        print("Auto-resume disabled")
    
    print("NBV Framework Demo")
    print("=" * 50)
    print(f"Device: {config['device']}")
    print(f"Mode: {args.mode}")
    print(f"Max meshes: {config['max_meshes']}")
    
    # 创建合成数据（如果需要）
    if args.create_data or not os.path.exists(config["synthetic_data_root"]):
        create_synthetic_data(config)
    
    # 设置模型
    vggt_wrapper, policy_network, renderer, loss_fn = setup_models(config)
    
    if args.mode in ["train", "all"]:
        # 设置数据加载器
        train_loader, val_loader = setup_data_loaders(config)
        
        # 训练
        trainer = train_nbv_policy(
            config, vggt_wrapper, policy_network, renderer, 
            loss_fn, train_loader, val_loader
        )

    if args.mode in ["eval", "all"]:
        # 评估
        run_evaluation(config, vggt_wrapper, policy_network, renderer)
    
    print("\nDemo completed successfully!")


if __name__ == "__main__":
    mp.set_start_method('spawn') 
    main()
