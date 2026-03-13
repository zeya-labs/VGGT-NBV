# Depth Anything 3 Integration Report

## 背景

本次工作目标：

1. 将 `ByteDance-Seed/Depth-Anything-3` 集成到 `nbv_framework`，作为与 `MapAnything` 同级的重建模型。
2. 将 DA3 接入 `experiments/mapanything_recon_webui`。
3. 下载 `DA3-BASE` 权重并完成真实重建验证。
4. 排查 DA3 在 WebUI 中“重建效果特别差”的问题。

日期：2026-03-09

## 文档说明

本文件里的操作记录分两类：

- `实际执行命令`：本次会话中直接在终端执行过的命令。
- `等价复现命令`：对应本次实际动作的可复现 shell 操作，主要用于描述 `apply_patch` 编辑、第三方源码导入等不适合逐字展开的步骤。

## 一、代码集成

### 1. 引入上游运行时代码

将 DA3 运行时源码以最小可运行子集形式放入：

- `third_party/Depth-Anything-3/src`
- `third_party/Depth-Anything-3/LICENSE`

主包内不复制上游实现，只保留封装调用逻辑。

等价复现命令：

```bash
git clone https://github.com/ByteDance-Seed/Depth-Anything-3 /tmp/Depth-Anything-3
```

```bash
mkdir -p /mnt/sdb/chenmohan/VGGT-NBV/third_party/Depth-Anything-3
cp -r /tmp/Depth-Anything-3/src /mnt/sdb/chenmohan/VGGT-NBV/third_party/Depth-Anything-3/
cp /tmp/Depth-Anything-3/LICENSE /mnt/sdb/chenmohan/VGGT-NBV/third_party/Depth-Anything-3/
```

实际执行命令：

```bash
rg --files third_party/Depth-Anything-3 | sort
```

说明：

- 本次实际动作是将已获取的上游仓库最小运行子集复制到 `third_party/Depth-Anything-3/`。
- 文档中的 `git clone + cp` 是对该动作的等价复现写法，便于审计。

### 2. 新增 DA3 scene encoder

新增：

- `nbv_framework/models/scene_encoder/depthanything3_encoder.py`

实现内容：

- `DepthAnything3Wrapper`
- `extract_scene_features(...)`
- `reconstruct_and_evaluate(...)`
- Hugging Face 本地 / 远端权重解析
- DA3 config 推断 `scene_feature_dim`
- 与现有 `MapAnythingWrapper` 对齐的接口

补充能力：

- 自动向 `sys.path` 注入 `third_party/Depth-Anything-3/src`
- 若环境里没有 `addict`，提供兼容 shim
- 支持读取 `config.json` 和 `model.safetensors`

实际操作类型：

- 使用 `apply_patch` 新建：
  - `nbv_framework/models/scene_encoder/depthanything3_encoder.py`
- 使用 `apply_patch` 修改：
  - `nbv_framework/models/scene_encoder/__init__.py`
  - `nbv_framework/models/__init__.py`
  - `nbv_framework/__init__.py`

等价复现命令：

```bash
touch nbv_framework/models/scene_encoder/depthanything3_encoder.py
```

```bash
$EDITOR nbv_framework/models/scene_encoder/depthanything3_encoder.py
$EDITOR nbv_framework/models/scene_encoder/__init__.py
$EDITOR nbv_framework/models/__init__.py
$EDITOR nbv_framework/__init__.py
```

### 3. 新增 adapter 并接入训练装配

新增：

- `nbv_framework/adapters/scene_encoder/depthanything3_adapter.py`

修改：

- `nbv_framework/training/factory.py`
- `nbv_framework/config/schema.py`
- `nbv_framework/config/validation.py`
- `configs/nbv/train.yaml`

接入结果：

- 新增 `model.scene_encoder_type`
- 支持 `mapanything` / `depthanything3`
- DA3 配置项包括：
  - `depthanything3_model_name_or_path`
  - `depthanything3_revision`
  - `depthanything3_local_files_only`
  - `depthanything3_feature_layer`
  - `depthanything3_use_ray_pose`
  - `depthanything3_ref_view_strategy`

实际操作类型：

- 使用 `apply_patch` 新建：
  - `nbv_framework/adapters/scene_encoder/depthanything3_adapter.py`
- 使用 `apply_patch` 修改：
  - `nbv_framework/adapters/scene_encoder/__init__.py`
  - `nbv_framework/adapters/__init__.py`
  - `nbv_framework/training/factory.py`
  - `nbv_framework/config/schema.py`
  - `nbv_framework/config/validation.py`
  - `configs/nbv/train.yaml`
  - `nbv_framework/tests/test_config_validation.py`

等价复现命令：

```bash
touch nbv_framework/adapters/scene_encoder/depthanything3_adapter.py
```

```bash
$EDITOR nbv_framework/adapters/scene_encoder/depthanything3_adapter.py
$EDITOR nbv_framework/adapters/scene_encoder/__init__.py
$EDITOR nbv_framework/adapters/__init__.py
$EDITOR nbv_framework/training/factory.py
$EDITOR nbv_framework/config/schema.py
$EDITOR nbv_framework/config/validation.py
$EDITOR configs/nbv/train.yaml
$EDITOR nbv_framework/tests/test_config_validation.py
```

### 4. 接入 WebUI

修改：

- `experiments/mapanything_recon_webui/backend/pipeline.py`
- `experiments/mapanything_recon_webui/backend/app.py`
- `experiments/mapanything_recon_webui/backend/schemas.py`
- `experiments/mapanything_recon_webui/frontend/index.html`
- `experiments/mapanything_recon_webui/frontend/app.js`
- `experiments/mapanything_recon_webui/README.md`

接入结果：

- WebUI 可切换 `MapAnything` / `Depth Anything 3`
- 历史记录保留 `reconstruction_model`
- `Depth Anything 3` 默认忽略 `depth_z`
- 导出的 PLY 文件名包含后端类型

实际操作类型：

- 使用 `apply_patch` 修改：
  - `experiments/mapanything_recon_webui/backend/pipeline.py`
  - `experiments/mapanything_recon_webui/backend/app.py`
  - `experiments/mapanything_recon_webui/backend/schemas.py`
  - `experiments/mapanything_recon_webui/frontend/index.html`
  - `experiments/mapanything_recon_webui/frontend/app.js`
  - `experiments/mapanything_recon_webui/README.md`

等价复现命令：

```bash
$EDITOR experiments/mapanything_recon_webui/backend/pipeline.py
$EDITOR experiments/mapanything_recon_webui/backend/app.py
$EDITOR experiments/mapanything_recon_webui/backend/schemas.py
$EDITOR experiments/mapanything_recon_webui/frontend/index.html
$EDITOR experiments/mapanything_recon_webui/frontend/app.js
$EDITOR experiments/mapanything_recon_webui/README.md
```

## 二、权重与运行时准备

### 1. DA3-BASE 权重

最终权重目录：

- `models/DepthAnything3/DA3-BASE/config.json`
- `models/DepthAnything3/DA3-BASE/model.safetensors`

说明：

- 远端 `snapshot_download(...)` 一度遇到 Hugging Face SSL EOF 问题。
- 之后确认本机 HF cache 已有完整 `model.safetensors`，再复制到项目目录。

实际执行命令：

```bash
source ./.venv/bin/activate && python - <<'PY'
from huggingface_hub import snapshot_download
path = snapshot_download(
    repo_id='depth-anything/DA3-BASE',
    allow_patterns=['config.json', 'model.safetensors'],
    local_dir='/mnt/sdb/chenmohan/VGGT-NBV/models/DepthAnything3/DA3-BASE',
    local_dir_use_symlinks=False,
    resume_download=True,
)
print(path)
PY
```

```bash
source ./.venv/bin/activate && python - <<'PY'
from huggingface_hub import try_to_load_from_cache
for filename in ['config.json', 'model.safetensors']:
    p = try_to_load_from_cache('depth-anything/DA3-BASE', filename)
    print(filename, p)
PY
```

```bash
cp -f /home/hanyufei/.cache/huggingface/hub/models--depth-anything--DA3-BASE/snapshots/f4a6c9b3c95e41c82048423d3493a81ec3fa810e/model.safetensors \
  /mnt/sdb/chenmohan/VGGT-NBV/models/DepthAnything3/DA3-BASE/model.safetensors
```

### 2. GPU / 环境确认

已确认：

- `torch.cuda.is_available() == True`
- 4 张 RTX 3090 可用

实际执行命令：

```bash
source ./.venv/bin/activate && CUDA_VISIBLE_DEVICES=0 python - <<'PY'
import torch
print('cuda_available', torch.cuda.is_available())
print('device_count', torch.cuda.device_count())
for i in range(torch.cuda.device_count()):
    props = torch.cuda.get_device_properties(i)
    print(i, props.name, props.total_memory)
PY
```

## 三、验证操作

### 1. 静态导入 / 配置验证

已执行：

- `python -m compileall nbv_framework experiments/mapanything_recon_webui/backend third_party/Depth-Anything-3/src`
- `python -m nbv_framework.scripts.train --cfg job`

实际执行命令：

```bash
python -m compileall nbv_framework experiments/mapanything_recon_webui/backend third_party/Depth-Anything-3/src
```

```bash
python -m nbv_framework.scripts.train --cfg job
```

验证结论：

- Hydra 配置正常解析
- 训练装配可识别 `depthanything3`
- WebUI backend 可导入

### 2. DA3 wrapper 单独烟雾测试

使用本地 `DA3-BASE` 权重在 GPU 上验证：

- `extract_scene_features` 输出形状：`(1, 2, 100, 768)`
- `reconstruct_and_evaluate` 输出形状：`(1, 2, 140, 140, 3)`

说明：

- DA3 能正常加载
- scene feature 与重建输出都已可用

实际执行命令：

```bash
source ./.venv/bin/activate && CUDA_VISIBLE_DEVICES=0 python - <<'PY'
import sys
import time
import torch

sys.path.insert(0, '/mnt/sdb/chenmohan/VGGT-NBV')
from nbv_framework.models.scene_encoder.depthanything3_encoder import DepthAnything3Wrapper

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('device', device)
start = time.time()
model = DepthAnything3Wrapper(
    model_name_or_path='/mnt/sdb/chenmohan/VGGT-NBV/models/DepthAnything3/DA3-BASE',
    local_files_only=True,
).to(device)
print('loaded_s', round(time.time() - start, 2))

images = torch.rand(1, 2, 3, 140, 140, device=device)
poses = torch.tensor([[
    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
    [0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
]], device=device, dtype=torch.float32)

with torch.inference_mode():
    feats, views = model.extract_scene_features(images, poses, fov_degrees=60.0)
    recon = model.reconstruct_and_evaluate(images, poses, fov_degrees=60.0)

print('feats', tuple(feats.shape))
print('views', len(views), tuple(views[0]['img'].shape))
print('recon_world_points', tuple(recon.recon_world_points.shape))
print('recon_conf', tuple(recon.recon_conf.shape))
print('recon_mask', tuple(recon.recon_mask.shape))
PY
```

### 3. WebUI 真实链路验证

实际跑通：

1. `GET /api/mesh_roots`
2. `GET /api/mesh_list`
3. `POST /api/prepare_inputs`
4. `POST /api/reconstruct`
5. `GET /`
6. `GET /results/...ply`

典型成功样例：

- `run_id=20260309_012748_e1e920`
- mesh: `models/House3K_obj/BATCH_7/SET_B/BAT7_SETB_HOUSE10_WTR.obj`

导出文件：

- `experiments/mapanything_recon_webui/results/20260309_012748_e1e920/reconstruction/point_cloud_depthanything3_both_without_depth.ply`

实际执行命令：

```bash
source /mnt/sdb/chenmohan/VGGT-NBV/.venv/bin/activate && \
export NBV_DA3_MODEL_NAME_OR_PATH=/mnt/sdb/chenmohan/VGGT-NBV/models/DepthAnything3/DA3-BASE && \
export NBV_DA3_LOCAL_FILES_ONLY=1 && \
uvicorn app:app --host 127.0.0.1 --port 8010
```

```bash
curl -s http://127.0.0.1:8010/api/mesh_roots
```

```bash
curl -s 'http://127.0.0.1:8010/api/mesh_list?root=models/House3K_obj&limit=5'
```

```bash
curl -s -X POST http://127.0.0.1:8010/api/prepare_inputs \
  -H 'Content-Type: application/json' \
  -d '{"mesh_path":"models/House3K_obj/BATCH_7/SET_B/BAT7_SETB_HOUSE10_WTR.obj","num_views":2,"render":{"image_size":140,"fov":60.0,"normalize_method":"unit_sphere","num_samples":32768},"sampling":{"view_sampling_mode":"deterministic_per_call","seed":42,"camera_radius":1.6,"camera_radius_variation":0.0,"camera_radius_mode":"random","up_axis":"Y","scene_index":0,"use_manual_camera":false},"show_depth":false}'
```

```bash
curl -s -X POST http://127.0.0.1:8010/api/reconstruct \
  -H 'Content-Type: application/json' \
  -d '{"run_id":"20260309_012748_e1e920","reconstruction_model":"depthanything3","conf_threshold":0.0,"max_points":100000,"use_depth_input":true,"display_mode":"both"}'
```

## 四、定位到的问题

### 现象

用户反馈通过以下命令启动的 WebUI 中，DA3 重建效果非常差：

```bash
python -m uvicorn app:app --host 0.0.0.0 --port 8000
```

### 原因

问题不是 “DA3 本身就差”，而是集成代码漏掉了 DA3 官方推理中的一个关键步骤：

- 使用 `output.extrinsics` 与输入位姿做 Umeyama scale 对齐
- 再用该尺度修正 `depth`

旧实现的问题：

1. 输入位姿先被归一化后送入 DA3。
2. 旧代码直接拿 `output.depth` 做回投。
3. 没有根据 DA3 预测位姿与输入位姿的尺度差修正深度。

结果：

- 点云尺度系统性错误
- WebUI 中看起来像 “DA3 重建质量很差”

复现与诊断时实际执行命令：

```bash
source /mnt/sdb/chenmohan/VGGT-NBV/.venv/bin/activate && \
export NBV_DA3_MODEL_NAME_OR_PATH=/mnt/sdb/chenmohan/VGGT-NBV/models/DepthAnything3/DA3-BASE && \
export NBV_DA3_LOCAL_FILES_ONLY=1 && \
python -m uvicorn app:app --host 0.0.0.0 --port 8000
```

```bash
source /mnt/sdb/chenmohan/VGGT-NBV/.venv/bin/activate && CUDA_VISIBLE_DEVICES=0 python - <<'PY'
# 这里实际执行的是一段内联 Python 脚本：
# 1. 用 prepare_inputs_for_run(...) 生成同一组输入
# 2. 分别计算“旧 wrapper 路径”和“按官方 Umeyama scale 对齐后的路径”
# 3. 输出 sym_current / sym_official_like / bbox_diag 等指标
PY
```

## 五、修复内容

修复文件：

- `nbv_framework/models/scene_encoder/depthanything3_encoder.py`

新增逻辑：

1. `_camera_centers_from_extrinsics(...)`
2. `_estimate_umeyama_scale(...)`
3. 在 `reconstruct_and_evaluate(...)` 中：
   - 读取 `output.extrinsics`
   - 估计输入位姿到 DA3 预测位姿的 Sim(3) 尺度
   - 用该尺度修正 `depth`
   - 在输入世界坐标系下回投

兼容处理：

- DA3 输出的 `extrinsics` 为 `3x4`
- 输入外参为 `4x4`
- 修复中已统一支持两种形式

实际操作类型：

- 使用 `apply_patch` 修改：
  - `nbv_framework/models/scene_encoder/depthanything3_encoder.py`

修复后执行的最小回归命令：

```bash
python -m compileall nbv_framework/models/scene_encoder/depthanything3_encoder.py
```

## 六、修复前后对比

使用同一组输入，计算 predicted point cloud 与 GT point cloud 的对称最近邻误差（越低越好）。

### Case 1

- mesh: `BAT1_SETA_HOUSE1`
- 修复前：`0.794`
- 修复后：`0.105`
- 改善倍数：约 `7.4x`

### Case 2

- mesh: `BAT1_SETC_HOUSE48`
- 修复前：`0.965`
- 修复后：`0.149`
- 改善倍数：约 `6.5x`

### Case 3

- mesh: `BAT7_SETB_HOUSE10_WTR`
- 修复前：`0.947`
- 修复后：`0.267`

结论：

- 这是集成代码问题，不是 DA3 本身能力上限。

用于得到上面数字的实际执行命令：

```bash
source /mnt/sdb/chenmohan/VGGT-NBV/.venv/bin/activate && CUDA_VISIBLE_DEVICES=0 python - <<'PY'
# 这里实际执行的是一段多样本对比脚本：
# - BAT1_SETA_HOUSE1
# - BAT1_SETC_HOUSE48
# 输出：
#   umeyama_scale
#   sym_current
#   sym_official_like
#   improve_ratio
PY
```

## 七、修复后再次验证 WebUI

已重新使用以下命令启动新版 backend：

```bash
source /mnt/sdb/chenmohan/VGGT-NBV/.venv/bin/activate && \
export NBV_DA3_MODEL_NAME_OR_PATH=/mnt/sdb/chenmohan/VGGT-NBV/models/DepthAnything3/DA3-BASE && \
export NBV_DA3_LOCAL_FILES_ONLY=1 && \
python -m uvicorn app:app --host 0.0.0.0 --port 8000
```

并重新完成真实重建：

- `run_id=20260309_014706_54d694`
- mesh: `models/House3K_obj/BATCH_1/Set_A/BAT1_SETA_HOUSE1.obj`

输出：

- `experiments/mapanything_recon_webui/results/20260309_014706_54d694/reconstruction/point_cloud_depthanything3_both_without_depth.ply`
- `experiments/mapanything_recon_webui/results/20260309_014706_54d694/reconstruction/reconstruct_metadata.json`

返回摘要：

- `num_points=61133`
- `num_points_gt=21933`
- `num_points_recon=39200`
- `used_depth_input=false`

实际执行命令：

```bash
curl -s -X POST http://127.0.0.1:8000/api/prepare_inputs \
  -H 'Content-Type: application/json' \
  -d '{"mesh_path":"models/House3K_obj/BATCH_1/Set_A/BAT1_SETA_HOUSE1.obj","num_views":2,"render":{"image_size":140,"fov":60.0,"normalize_method":"unit_sphere","num_samples":32768},"sampling":{"view_sampling_mode":"deterministic_per_call","seed":42,"camera_radius":1.6,"camera_radius_variation":0.0,"camera_radius_mode":"random","up_axis":"Y","scene_index":0,"use_manual_camera":false},"show_depth":false}'
```

```bash
curl -s -X POST http://127.0.0.1:8000/api/reconstruct \
  -H 'Content-Type: application/json' \
  -d '{"run_id":"20260309_014706_54d694","reconstruction_model":"depthanything3","conf_threshold":0.0,"max_points":100000,"use_depth_input":true,"display_mode":"both"}'
```

## 八、当前已知事项

### 1. DA3 仍忽略 depth_z

当前 DA3 路径不会消费 `depth_z`，这与 WebUI 返回的：

- `used_depth_input=false`

一致。

### 2. 输入图像黑底未做前景抠除

当前逻辑：

- 输入给 DA3 的 RGB 保留渲染黑底
- 最终导出的 GT / Recon 点云会按 mask 去掉背景点

### 3. 加载 checkpoint 时仍有 missing keys 警告

目前会看到：

- `output_conv2_aux.*` 的少量 missing keys

现状：

- 不影响当前 `extract_scene_features(...)`
- 不影响当前 `reconstruct_and_evaluate(...)`
- 真实推理已通过

## 九、后续建议

建议后续继续做两件事：

1. 将 DA3 官方 ambiguity / edge mask 接入 `recon_mask`，进一步清理红色点云噪声。
2. 在 WebUI 中增加一个“DA3 官方 scale alignment 已启用”的 debug 信息，避免后续再次误判为模型本身问题。

## 十、审计附录

用于全面 review 当前工作区改动的实际执行命令：

```bash
git status --short
```

```bash
rg --files third_party/Depth-Anything-3 | sort
```
