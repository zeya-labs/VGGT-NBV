# 仓库指南（Repository Guidelines）

## 项目结构与模块组织
- `nbv_framework/`：NBV 主流程代码（数据、模型、渲染、训练、工具函数）。
- `configs/nbv/train.yaml`：根训练/评估入口配置（Hydra）。
- `train.py`：Lightning + Hydra 主入口（`mode=train|test|train_test`）。
- `experiments/`：实验脚本与原型工具（含 `experiments/nbv_webui/`）。
- `models/` 与 `runs/`：本地资源与实验产物目录。
- `map-anything/`、`vggt/`：Git 子模块；更新时应明确提交子模块指针变更。

## 构建、测试与开发命令
```bash
# 激活本地虚拟环境
source ./.venv/bin/activate

# 4 卡分布式训练（默认配置）
torchrun --nproc_per_node=4 train.py mode=train

# 从指定 checkpoint 执行评估
torchrun --nproc_per_node=4 train.py mode=test resume_checkpoint=/abs/path/model.ckpt

# 在子模块中执行格式化与静态检查
cd map-anything && pre-commit run -a

# 可选：运行 MapAnything 测试
cd map-anything && pytest
```
本地快速验证可用 Hydra 覆盖参数，例如：`batch_size=1 max_meshes=10 trainer.devices=1`。

## 代码风格与命名规范
- Python 使用 4 空格缩进；新增/修改的公共接口建议补充类型标注。
- 命名规则：模块/函数/变量使用 `snake_case`，类名使用 `PascalCase`，Hydra 键名保持小写。
- 代码归位：训练逻辑放 `nbv_framework/training/`，几何放 `nbv_framework/geometry/`，渲染放 `nbv_framework/rendering/`。
- `map-anything/` 代码遵循 Ruff 钩子：`ruff format`、`ruff check --fix`。

## 测试指南
- 仓库未配置统一覆盖率门槛；每个修复或功能变更都应补充针对性测试。
- 优先编写可复现的单元测试（工具函数/变换逻辑），训练路径改动至少做一次烟雾测试。
- 测试命名建议使用 `test_*.py` 或 `*_test.py`，并尽量与被测模块同目录维护。

## 提交与 Pull Request 规范
- 优先使用 Conventional Commit：如 `feat(nbv): ...`、`fix: ...`、`refactor(...)`、`docs(...)`、`perf(...)`。
- 单次提交聚焦单一变更；提交说明中写明关键 Hydra 覆盖参数与运行前提。
- PR 必须包含：变更目的/范围、验证步骤、前后指标对比（或界面截图）、关联 issue/任务。
- 涉及 `map-anything/` 或 `vggt/` 时，需明确说明是修改子模块代码还是仅更新子模块指针。
