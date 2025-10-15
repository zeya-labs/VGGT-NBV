# MapAnything 推理流程导读

本文梳理 NBV 框架中 MapAnything 的整套推理链路，帮助定位关键代码入口。推荐从高到低逐步阅读如下文件：

## 1. 框架入口：`nbv_framework/models/mapanything_wrapper.py`

- **场景特征提取**：`extract_scene_features` 会调用 `_prepare_views` 进行归一化与视角打包，然后通过 `MapAnything._encode_n_views` / `_encode_and_fuse_optional_geometric_inputs` 生成 `[B, S, P, D]` token，并在 `_update_feature_dim` 中记录真实维度。见 `nbv_framework/models/mapanything_wrapper.py:122-148`。
- **完整推理**：`reconstruct_and_evaluate` 组装 `views` 后直接调用 `MapAnything.forward`，并把 HuggingFace 模型原始输出映射到 NBV 框架所需的键（`world_points_from_depth`、`depth_conf` 等）。见 `nbv_framework/models/mapanything_wrapper.py:151-307`。
- **维度自适应**：`infer_feature_dim` 用零样本生成器探测输出宽度，并暴露 `feature_dim` 属性供策略网络使用。见 `nbv_framework/models/mapanything_wrapper.py:217-253`。

> **阅读建议**：先理解 `_prepare_views` 如何将 `[B, S, 3, H, W]` 正规化成 MapAnything 接受的字典，再追踪 `extract_scene_features` 和 `reconstruct_and_evaluate` 分别走向特征/完整推理两条路径。

## 2. 核心模型：`map-anything/mapanything/models/mapanything/model.py`

### 2.1 编码阶段
- `_encode_n_views`：拼接多视图，构造 `ViTEncoderInput`，调用 DINOv2 encoder 输出每视角 latent。见 `model.py:622-645`。
- `_encode_and_fuse_optional_geometric_inputs`：根据配置注入射线、深度、位姿等几何模态，最终通过 LayerNorm 融合。内部依次调用 `_encode_and_fuse_ray_dirs`、`_encode_and_fuse_depths`、`_encode_and_fuse_cam_quats_and_trans`。见 `model.py:1133-1260`。

### 2.2 共享与预测
- `forward` 主体分 6 步：特征编码 → 几何融合 → Scale Token 扩展 → 交替注意力 Transformer (`info_sharing`) → 下游头部 (`downstream_head`) → 根据 `scene_rep_type` 组装输出。见 `model.py:1477-1760`。
- `downstream_head` 根据配置选择线性/DPT 头，分别产出稠密几何、位姿和尺度通道。见 `model.py:1262-1475`。
- `scene_rep_type` 分支决定最终返回的键（`pts3d`、`ray_directions`、`cam_trans` 等）。可查 `model.py:1618-1759`。

### 2.3 推理封装
- `infer`：面向用户的高阶接口，实现严格的输入校验、自动混合精度、后处理调用。见 `model.py:1964-2050`。
- `_configure_geometric_input_config`：控制图像/几何多模态的启用与采样概率。见 `model.py:1911-1954`。

## 3. 输入/输出预处理：`map-anything/mapanything/utils/inference.py`

- `validate_input_views_for_inference`：检查视图字典是否包含必需键、是否有冲突。见 `utils/inference.py:128-200`。
- `preprocess_input_views_for_inference`：将相机内参转成射线、`depth_z` 转 `depth_along_ray`、位姿转换为四元数+平移，并补齐 `is_metric_scale`。见 `utils/inference.py:203-292`。
- `postprocess_model_outputs_for_inference`：把模型 raw 输出添补反归一化图像、内参、位姿矩阵和掩码，同时可按需做边缘/置信度过滤。见 `utils/inference.py:295-380`。

## 4. 推荐阅读顺序

1. **Wrapper** (`nbv_framework/models/mapanything_wrapper.py`)：了解 NBV 框架如何准备输入、调用 MapAnything，并将结果适配到现有损失。
2. **模型 forward** (`mapanything/models/mapanything/model.py`)：掌握 Encoder → Transformer → 头部的内部流程及可选分支。
3. **Infer 工具** (`mapanything/utils/inference.py`)：最后补充输入校验、后处理细节，方便调通自定义数据或分析输出。

掌握以上三个层级后，即可完整跟踪一次 `MapAnythingWrapper.reconstruct_and_evaluate` 调用中发生的所有步骤。

