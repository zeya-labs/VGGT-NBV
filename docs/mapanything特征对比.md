# MapAnything 与 VGGT 特征对比说明

> 本文档总结了 NBV 框架中 MapAnythingWrapper 与 VGGTWrapper 在特征抽取路径上的差异，帮助迁移策略网络或调试相关模块。

## 1. 特征生成流程对比

### VGGTWrapper
- 调用 `self.vggt_model.aggregator` 得到 24 层 token 序列，每层形状固定为 `[B, S, P, 2048]`，并保留相机 token / register token 的语义划分。见 `nbv_framework/models/vggt_wrapper.py:141`。
- `extract_scene_features` 默认返回最后一层的 token，未做额外 reshape，策略网络可按相机 / patch 分段处理。见 `nbv_framework/models/vggt_wrapper.py:162-209`。
- 输出维度固定 2048，并假定 token 顺序为 `[camera, register, patches]`。

### MapAnythingWrapper
- 通过 `MapAnything._encode_n_views` 和 `_encode_and_fuse_optional_geometric_inputs` 获得每视角的稠密特征图，再进入多视角 Transformer。见 `nbv_framework/models/mapanything_wrapper.py:124-148`。
- `_gather_tokens` 会将 `[B, C, H, W]` 或 `[B, tokens, D]` 的特征统一展开为 `[B, tokens, D]`，并按视角堆叠为 `[B, S, P, D]`，但**不包含 VGGT 的相机 / register token 语义**。见 `nbv_framework/models/mapanything_wrapper.py:301-317`。
- 特征维度取决于 MapAnything 配置（默认 dinov2-large 为 1024），后转768

## 2. Token 语义差异

- VGGT token 顺序明确：索引 0 为相机 token，索引 1-4 为 register token，之后是 patch token，可用 `token_type` 选择。见 `nbv_framework/models/vggt_wrapper.py:196-206`。
- MapAnything token 来源于稠密空间特征，未区分相机 / register。若策略网络依赖 VGGT 特殊 token（例如只取 camera token），需要改写为整体池化或自定义聚合。

## 3. 维度与归一化

- VGGT 固定 2048 维，无需额外设定。
- MapAnythingWrapper 在初始化时读取 `enc_embed_dim` 并且在 `train.py` 中对策略网络输入维度做“auto”适配，避免硬编码。见 `train.py:110-117`。
- MapAnythingWrapper 借鉴 `load_images` 逻辑，对输入图像执行统一的裁剪、重采样与 DINO 风格归一化。见 `nbv_framework/models/mapanything_wrapper.py:194-259`。VGGT 依赖外部 `load_and_preprocess_images` 完成预处理，两者分布不同。

## 4. 推理输出映射

- 为适配现有损失函数，MapAnythingWrapper 将 `pts3d` 同步映射到 `world_points` / `world_points_from_depth`，并将 `conf` 复用为 `depth_conf`、`world_points_conf`。见 `nbv_framework/models/mapanything_wrapper.py:320-343`。
- VGGTWrapper 原生输出 `world_points` / `world_points_conf` 等字段，不需额外映射。见 `nbv_framework/models/vggt_wrapper.py:223-248`。

## 5. 使用建议

1. 策略网络请使用 `MapAnythingWrapper.feature_dim` 作为输入维度；`train.py:99-118` 会先调用 `MapAnythingWrapper.infer_feature_dim(...)` 做一次零样本推理，保证维度与真实输出一致。
2. 若依赖相机 token，请改成对全部 token 做池化或自定义聚合逻辑，可参考 `BaseNBVPolicy._pool_tokens_if_needed` 中的 `token_pooling_mode` 配置。
3. 评估或可视化时，注意 MapAnything 输出的是 metric reconstruction，数据范围与 VGGT 略有不同，必要时重新设定阈值。
