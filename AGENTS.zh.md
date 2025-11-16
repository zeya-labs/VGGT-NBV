# 仓库指南（中文版）

## 项目结构与模块组织
- 生产代码集中在 `nbv_framework/`：`models/`（各类策略网络 + VGGT 封装）、`rendering/`（PyTorch3D 渲染器）、`training/`（训练器、损失、日志工具）、`datasets/`（House3K 与合成数据加载器/采样器）以及 `utils/`（相机、评估、可视化工具）。新增框架代码都应落在这些子模块中。
- `experiments/` 用于受控实验，例如 `view1_camera_path_experiment.py` 或 Chamfer 扫描。这里是想法孵化区，验证稳定后请上游合并进 `nbv_framework/`。
- `docs/` 保存 MapAnything 说明与日志指南。一旦工作流或配置开关变化，请更新/新增对应文档，确保团队可复现。
- `test/` 目前存放复现案例与诊断脚本。后续自动化覆盖请放在单独的 `tests/` 目录，并使用 `test_<module>.py` 命名。
- 供应商与预训练内容位于 `vggt/`。根目录 `models/` 下的几何资产是只读输入，禁止修改。
- 产出物写入 `checkpoints/`，流式日志与 TensorBoard 事件写入 `runs/`，临时导出可用 `outputs/`。务必在提交中忽略这些目录。

## 构建、测试与开发命令
1. 激活共享环境：`conda activate /mnt/sdb/chenmohan/env/mapanything/`。
2. 标准训练依赖 Hydra 覆写：`python train.py mode=train create_data=true`（需要时追加一份小型合成数据集），或通过 `python train.py mode=train resume_checkpoint=/path/to/ckpt` 从断点恢复。
3. 评估/回归：`python train.py mode=eval resume_checkpoint=/path/to/ckpt`；如需一次跑完训练+评估，使用 `python train.py mode=all auto_resume=false`。
4. 多 GPU / 分布式任务走内置 `init_distributed_mode`：`torchrun --nproc_per_node=<gpus> train.py mode=train auto_resume=false`。
5. 长时间任务前务必用 `nvidia-smi` 确认 CUDA，可优先使用 `docs/日志使用指南.md` 中记录的驱动/工具链组合。
6. 探索性扫描或消融请放在 `experiments/`，以免影响主训练脚本。

## 编码风格与命名约定
- 遵循 PEP 8，使用四空格缩进，并为对外函数/类提供显式类型注解。配置字典统一 snake_case（如 `policy_hidden_dim`、`synthetic_data_root`），不要依赖全局变量。
- 新的策略或渲染器实现请继承现有抽象（`BaseNBVPolicy`、`DifferentiableRenderer`），保持训练器接口一致。
- 临时 Helper 或 CLI Demo 请以前缀 `demo_` 命名或放入 `experiments/`；若成熟度够高，请迁移到相应 `nbv_framework` 子模块并补充 docstring。
- 在难以一眼看懂的数学/同步逻辑（如相机归一化、分布式 barrier）附近增加简洁注释，降低新人上手成本。

## 测试与实验记录
- 使用 `train.py` 中的 `set_random_seed` 并在 PR 描述里记录具体 seed 与命令。默认期望使用 GPU 进行集成测试。
- 扩展数据集时，运行一次 loader dry run，并通过 `python train.py mode=all auto_resume=false create_data=true` 采集指标；把 `runs/<experiment>` 的日志片段附到评审里。
- 复用内建评估辅助函数（`evaluate_nbv_policy`、`compare_with_baselines`），若有自定义基线或偏差，需在 `docs/` 里记录，方便复现。
- `experiments/` 中的脚本要输出可复现的统计（Chamfer、覆盖度等），并带时间戳写入 `outputs/`。分享结果时引用这些日志。
- 未来自动化测试请放进 `tests/`（pytest 风格）；现有 `test/` 目录继续保留人工调试用例。

## 分支、提交与 PR 流程
- 开发前必须新建分支，不得直接在 `develop` 上提交。分支命名推荐 `feature/<scope>-<desc>` 或 `bugfix/<issue>-<summary>`，方便评审追踪。
- 遵循 Conventional Commits（如 `feat(camera): add hemispheric sampler`），提交标题最长 72 字符。合并前请本地整理零碎 WIP 提交。
- PR 需关联相关 issue/任务，说明依赖的资产或 checkpoint，并附上关键日志或 TensorBoard 截图。凡新增环境变量或配置选项，都要列表说明。
- 禁止提交生成数据（`models/synthetic_data/`、`outputs/`、`runs/`、`checkpoints/`）。大体量资产请通过约定的文件渠道共享，并在 PR 中附链接。

## 安全与配置提示
- 几何资产或衍生数据一律不得入库。共享 checkpoint 前要清理本地路径、机器信息等敏感元数据。
- 新增的环境变量、密钥或服务入口需写入 `docs/` 并在 PR 中引用，确保团队能同步配置且不暴露敏感内容。
