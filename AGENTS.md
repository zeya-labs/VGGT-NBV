# 仓库指南（Repository Guidelines）

## 项目结构与模块组织
- `nbv_framework/`：主代码包，当前以训练流水线为主轴组织。
- `nbv_framework/cli/`：CLI 入口；`nbv_framework.cli.train:main` 是当前 Hydra 训练入口。
- `nbv_framework/training/`：Lightning module、datamodule、trainer、losses、装配工厂。
- `nbv_framework/workflows/`：训练步骤编排与用例逻辑。
- `nbv_framework/data/`：数据集、loader、collate、House3K 相关数据逻辑。
- `nbv_framework/models/`：策略网络与 scene encoder。
- `nbv_framework/geometry/`：几何与位姿运算。
- `nbv_framework/reconstruction/`：重建数据结构与重建辅助逻辑。
- `nbv_framework/adapters/` 与 `nbv_framework/ports/`：外部依赖适配层与 Protocol 接口约定。
- `nbv_framework/infrastructure/`：保留低层实现与通用能力，目前主要是 `rendering/`、`observability/`、`utils/`。
- `configs/nbv/train.yaml`：根训练/评估配置。
- `train.py`：仓库根训练入口，直接转发到 `nbv_framework.cli.train`。
- `third_party/`：外来源码；当前包含 `Density_aware_Chamfer_Distance/`，不要与主包代码混放。
- `models/` 与 `outputs/`：本地资源与实验产物。
- `map-anything/`、`vggt/`：Git 子模块；修改时需明确是改子模块源码还是仅更新子模块指针。

## 构建、测试与开发命令
```bash
# 激活本地虚拟环境
source ./.venv/bin/activate

# 仅检查 Hydra 配置是否可解析
python train.py --cfg job

# 4 卡分布式训练
torchrun --nproc_per_node=4 train.py

# 从指定 checkpoint 执行评估
torchrun --nproc_per_node=4 train.py \
  experiment.mode=test \
  experiment.resume_checkpoint=/abs/path/model.ckpt

# 训练后立刻测试
torchrun --nproc_per_node=4 train.py \
  experiment.mode=train_test \
  experiment.resume_checkpoint=/abs/path/model.ckpt

# 快速单机烟雾验证
python train.py \
  data.batch_size=1 \
  data.max_meshes=10 \
  runtime.trainer.devices=1 \
  runtime.trainer.limit_val_batches=0

# 子模块格式化与检查
cd map-anything && pre-commit run -a

# 可选：若环境已安装 pytest，执行针对性测试
python -m pytest nbv_framework/tests
```

## 代码归位约定
- 新训练逻辑优先放入 `nbv_framework/training/` 或 `nbv_framework/workflows/`，不要再引入新的 `application/`、`domain/`、`interfaces/` 风格目录。
- 数据逻辑放 `nbv_framework/data/`；模型放 `nbv_framework/models/`；几何放 `nbv_framework/geometry/`。
- 外部系统边界通过 `nbv_framework/ports/` 与 `nbv_framework/adapters/` 表达；`Port` 是 `Protocol`，不是运行时父类。
- 第三方代码只能放 `third_party/`；主包内只保留薄封装或调用逻辑。
- 若改动 `nbv_framework/infrastructure/rendering/`，同步检查依赖它的 adapter 与 workflow。

## 代码风格与命名规范
- Python 使用 4 空格缩进；新增或修改的公共接口尽量补充类型标注。
- 模块、函数、变量使用 `snake_case`，类名使用 `PascalCase`。
- Hydra 键名保持小写点路径风格，例如 `experiment.mode`、`runtime.trainer.devices`。
- 避免新增空壳包、转发层和“为架构而架构”的目录。
- 仓库内不要提交 `__pycache__/`、`*.pyc`、`dist/`、`build/`、`*.egg-info/`、日志文件或运行产物。

## 测试指南
- 仓库没有统一覆盖率门槛；每次修复或功能变更都应补充针对性验证。
- 优先写可复现的单元测试，尤其是几何、数据变换、workflow 编排和 loss 相关逻辑。
- 训练入口或装配层改动，至少做一次 `python train.py --cfg job` 和一次单机烟雾验证。
- 若环境中没有 `pytest`，至少执行 `python -m compileall nbv_framework third_party` 作为导入回归检查。
- 测试命名遵循 `test_*.py` 或 `*_test.py`，尽量与被测模块靠近维护。

## 提交与 Pull Request 规范
- 优先使用 Conventional Commit，例如 `feat(nbv): ...`、`fix: ...`、`refactor(nbv): ...`、`docs: ...`。
- 单次提交聚焦单一变更；提交说明中写明关键 Hydra 覆盖参数和运行前提。
- PR 至少包含：变更目的、影响范围、验证步骤、相关 issue/任务。
- 涉及 `third_party/`、`map-anything/` 或 `vggt/` 时，明确说明是修改第三方源码、修改子模块源码，还是仅更新指针。
