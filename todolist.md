# TODO：调整GT点云与掩码生成链路

## 背景
- 目前`nbv_framework/training/loss/reconstruction.py`中的`_render_gt_point_maps`会在前向计算时根据`gt_mesh`与`combined_camera_poses`即时渲染GT点云及有效掩码。
- 该逻辑与数据准备耦合紧密，导致损失函数承担了数据整理职责，也增加了训练阶段的渲染开销。
- 需求是把GT点云与mask的生成提前到数据集侧（`BaseDataset`/`House3KDataset`），使得`DataLoader`返回的`batch`直接包含`gt_points`与`valid_mask`，损失函数仅消费现成数据。

## 修改目标
1. 数据集产出结构中增加按视图展开的GT点云(`gt_point_maps`)及有效掩码(`gt_valid_masks`)，必要时还能附带压缩的点云列表。
2. 保证所有依赖（训练、评估、可视化）均改用新字段，不再在`ReconstructionLoss`内部触发渲染。
3. 维持`House3KDataset`现有渲染流程（用于生成初始图像），并在该流程里同步产出GT点云数据；其他数据集如暂未实现渲染需提供兼容分支或显式报错。

## 详细代办事项
1. **抽取通用渲染工具**  
   - 将`ReconstructionLoss._render_gt_point_maps`迁移到`nbv_framework/utils/mesh_utils.py`或新建`nbv_framework/utils/render_utils.py`，封装为`render_gt_point_maps(mesh: Meshes, camera_poses: torch.Tensor, renderer: DifferentiableRenderer)`，返回(`point_maps`, `valid_masks`)。  
   - 保留写TensorBoard的接口可选参数，便于训练阶段复用。  
   - 确认设备迁移逻辑：渲染使用renderer.device，返回后转CPU方便被`DataLoader`拼接。

2. **扩展`BaseDataset`接口**  
   - 在`BaseDataset.__getitem__`的通用实现中加入可选流程：当`camera_poses`有值且派生类声明支持渲染时，调用新工具生成`gt_point_maps`与`gt_valid_masks`。  
   - 新增抽象/虚方法（如`supports_gt_point_maps`或`get_renderer`），让不支持即时渲染的子类返回`False`，此时显式跳过并在`batch`中注入`None`占位，便于上游判断。  
   - 考虑数据集多进程加载：渲染器初始化需懒加载并放在工作进程上下文中，避免重复创建GPU资源。必要时新增缓存字典。

3. **修改`House3KDataset`**  
   - 复用现有`_render_images_from_mesh_data`前的renderer，生成图像时同步收集GT点云：在确定`selected_camera_poses`后，一次调用通用工具获得`gt_point_maps`/`valid_masks`。  
   - 将结果写入返回字典，例如`result["gt_point_maps"]`（形状`[N, H, W, 3]`）与`result["gt_valid_masks"]`（形状`[N, H, W]`，布尔或uint8）。  
   - 如果House3K还需返回按视图采样的稀疏点云（如`gt_correspondence_points`），在同一位置构建，便于后续Chamfer处理。

4. **更新自定义`collate_fn`**  
   - 在`nbv_framework/datasets/collate_functions.py`中为`gt_point_maps`和`gt_valid_masks`添加聚合逻辑，保持与`gt_mesh_data`类似的处理方式（堆叠成`torch.Tensor`）。  
   - 确保数据类型与维度在batch拼接后仍正确（`float32`与`bool`/`uint8`），并兼容值为`None`的情况。

5. **调整训练流程**  
   - 在`trainer.py`中，构造`combined_camera_poses`后需同步扩展GT点云：
     - 若新增视图来自策略网络，先调用通用工具生成对应的GT点云与mask，再与batch中已有的`gt_point_maps`/`gt_valid_masks`拼接。  
     - 在传递给损失函数前，将这些扩展结果打包到`gt_mesh_data`或独立字段（如`gt_correspondences`），保证与`recon_data`对齐。  
   - 若训练过程需记录GT点云（用于可视化），沿用原有日志目录结构。

6. **重构`ReconstructionLoss`**  
   - 移除`_render_gt_point_maps`及相关renderer依赖，仅保留对`gt_data`中`gt_point_maps`/`gt_valid_masks`的读取与shape校验。  
   - `extract_point_cloud_from_reconstruction`继续输出`correspondence_mask`，使用来自batch的`gt_valid_masks`进行交集过滤。  
   - 调整Chamfer计算路径：
     - 通过`correspondence_mask`从`gt_point_maps`中取出逐像素GT点组成`correspondence_points`列表。
     - 如`gt_point_maps`缺失（None），在日志中给出告警并跳过Chamfer或回落到旧路径（可选）。

7. **同步评估与可视化脚本**  
   - 检查`nbv_framework/utils/evaluation.py`等使用`gt_mesh_data`的模块，改为读取新的字段。  
   - 确认导出/保存时能够正确处理新增tensor（尤其是保存在CPU上避免GPU内存泄漏）。

8. **设备与内存管理**  
   - 明确在数据加载阶段返回的`gt_point_maps`/`gt_valid_masks`默认为CPU张量，在训练循环进入GPU前（`trainer.training_step`）统一搬运，避免不必要的显存占用。  
   - 对大分辨率场景评估内存压力，必要时提供采样或压缩策略（例如按固定网格下采样或仅保留mask）。

9. **测试与验证**  
   - 增加最小化脚本或单元测试，加载`House3KDataset`单个样本，检查返回字典是否包含新字段且尺寸正确。  
   - 运行一次`train.py --mode train --max_steps 1`验证训练循环能正常前向、Chamfer损失有数值。  
   - 若提供回退路径，模拟无renderer环境确保不会崩溃。

10. **文档与代码注释**  
    - 在数据集与损失函数关键处补充docstring，说明GT点云生成提前的原因及数据格式。  
    - 如需新增配置开关（例如`prefetch_gt_points`），更新README或相关配置说明。

> 注意：在移动渲染逻辑时，确认不会引入竞态（多进程dataloader）及重复CUDA上下文；必要时在`BaseDataset`中使用懒加载+线程锁。