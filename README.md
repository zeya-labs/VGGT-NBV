# NBV Framework: 目标驱动的策略学习框架

**基于基础模型监督的主动三维重建**

## 项目概述

本项目实现了一种全新的Next-Best-View (NBV)策略学习范式，通过直接优化最终重建质量来训练NBV决策网络，从根本上解决了传统代理指标的"目标错配"问题。

### 核心思想

- **目标驱动**：摒弃传统代理指标，直接以重建质量为优化目标
- **基础模型监督**：利用冻结的VGGT模型提供监督信号
- **端到端学习**：可微分的训练框架，支持梯度回传
- **策略涌现**：智能探索策略从目标优化中自发涌现

## 架构设计

```
NBV Framework
├── models/
│   ├── vggt_wrapper.py      # VGGT基础模型封装
│   └── policy_network.py    # NBV策略网络
├── rendering/
│   └── differentiable_renderer.py  # 可微分渲染器
├── training/
│   ├── trainer.py           # 训练框架
│   └── losses.py           # 损失函数
└── utils/
    ├── data_utils.py       # 数据处理
    ├── visualization.py    # 可视化工具
    └── evaluation.py       # 评估工具
```

## 核心组件

### 1. VGGTWrapper - 冻结的基础模型
- **场景编码器**：从多视图提取高维场景特征
- **质量评估器**：生成三维重建并评估质量
- 所有参数冻结，仅作为特征提取和监督信号生成

### 2. NBVPolicyNetwork - 可训练策略网络
- 轻量级网络，接收场景特征输出相机位姿
- 支持球坐标和笛卡尔坐标输出模式
- 可选的注意力机制增强版本

### 3. DifferentiableRenderer - 可微分渲染
- 基于PyTorch3D的高质量渲染
- 从相机位姿和GT mesh生成新视图
- 支持批量渲染和梯度回传

### 4. NBVTrainer - 端到端训练框架
- 完整的目标驱动训练流程
- 自动化的状态编码→动作提议→环境交互→质量评估→策略更新
- 支持多种损失函数和正则化技术

## 安装与使用

### 环境要求
- Python >= 3.8
- PyTorch >= 2.0.0
- PyTorch3D >= 0.7.0
- CUDA (推荐)

### 安装依赖
```bash
pip install -r requirements.txt
```

### 快速开始

1. **运行演示脚本**：
```bash
python demo_nbv_framework.py --mode all --create_data
```

2. **仅训练模式**：
```bash
python demo_nbv_framework.py --mode train --create_data
```

3. **仅推理模式**：
```bash
python demo_nbv_framework.py --mode inference
```

4. **仅评估模式**：
```bash
python demo_nbv_framework.py --mode eval
```

### 自定义使用

```python
from nbv_framework import VGGTWrapper, NBVPolicyNetwork, DifferentiableRenderer, NBVTrainer

# 设置模型
vggt_wrapper = VGGTWrapper("facebook/VGGT-1B")
policy_network = NBVPolicyNetwork(scene_feature_dim=9, hidden_dim=256)
renderer = DifferentiableRenderer(image_size=518)

# 创建训练器
trainer = NBVTrainer(vggt_wrapper, policy_network, renderer, loss_fn)

# 训练
trainer.train(train_loader, val_loader, num_epochs=100)
```

## 训练流程

1. **状态编码**：VGGT从当前N个视图提取场景特征
2. **动作提议**：策略网络输出下一个相机位姿
3. **环境交互**：可微分渲染器生成新观测图像
4. **质量评估**：VGGT重建并计算几何损失
5. **策略更新**：反向传播更新策略网络参数

## 评估指标

- **重建质量**：基于Chamfer距离的几何精度
- **覆盖度改进**：新视角对场景覆盖的贡献
- **视角效率**：单位视角的质量提升
- **推理速度**：策略网络的计算效率

## 实验结果

框架支持与多种基线方法的比较：
- 随机采样
- 基于边界的探索
- 基于熵的不确定性最小化
- 学习的NBV策略

## 文件结构

```
.
├── nbv_framework/           # 核心框架代码
├── vggt/                   # VGGT原始代码
├── demo_nbv_framework.py   # 演示脚本
├── requirements.txt        # 依赖列表
└── README.md              # 本文件
```

## 技术特点

- **可扩展性**：模块化设计，易于添加新的策略网络和损失函数
- **效率性**：轻量级策略网络，快速推理
- **鲁棒性**：在多样化物体上训练，具有良好泛化能力
- **可解释性**：可视化工具帮助理解策略行为

## 未来工作

- [ ] 支持更多3D表示形式（NeRF、Gaussian Splatting等）
- [ ] 实现在线学习和自适应策略
- [ ] 集成更多基线方法进行比较
- [ ] 支持多物体场景的协同重建
- [ ] 优化大规模数据集的训练效率

## 引用

如果本项目对您的研究有帮助，请考虑引用：

```bibtex
@article{nbv_framework_2025,
  title={Objective-Driven Policy Learning for Active 3D Reconstruction via Foundation Model-based Supervision},
  author={NBV Research Team},
  year={2025}
}
```

## 许可证

本项目基于MIT许可证开源。详见LICENSE文件。

## 联系方式

如有问题或建议，请通过GitHub Issues联系我们。