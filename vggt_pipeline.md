### VGGT 完整流水线

**1. 输入 (Input)**
VGGT接收一个包含$N$张RGB图像的序列$I = (I_i)_{i=1}^N$作为输入，其中每张图像$I_i$的大小为$R^{3 \times H \times W}$。这些图像观察的是同一个3D场景。

**2. 特征骨干网络 (Feature Backbone - Transformer)**
这是VGGT的核心部分，一个大型Transformer网络。
*   **图像分块 (Patchification)**: 首先，每张输入的图像$I_i$被DINO 模型分块，并转换为一组$K$个图像令牌（tokens）$t_I^i \in R^{K \times C'}$。这些令牌包含了图像的局部视觉信息。
*   **令牌增强 (Token Augmentation)**: 对于每张图像$I_i$，除了其图像令牌$t_I^i$之外，还会额外添加一个“相机令牌”$t_c^i \in R^{1 \times C'}$和四个“注册令牌”$t_r^i \in R^{4 \times C'}$。这些令牌的串联$(t_I^i, t_c^i, t_r^i)$随后被送入Alternating-Attention Transformer。
    *   **参考帧识别**: 第一个图像的相机令牌$t_c^1$和注册令牌$t_r^1$被设置为一组与所有其他帧不同的可学习令牌，这使得模型能够区分第一帧作为世界参考坐标系，并在此坐标系下进行3D预测。
*   **交替注意力 (Alternating-Attention)**: Transformer的主网络结构采用了一种“交替注意力”（Alternating-Attention, AA）机制。它交替进行“帧内自注意力”（frame-wise self-attention）和“全局自注意力”（global self-attention）层。
    *   **帧内自注意力**: 作用于每帧内部的令牌，使得模型能专注于单个图像的内部信息。
    *   **全局自注意力**: 作用于所有帧的令牌，使得模型能够整合跨不同图像的信息。
    *   通过这种机制，Transformer在整合跨图像信息和规范化每张图像内部激活之间取得了平衡。
*   **骨干网络输出**: Transformer处理这些令牌后，输出相应的输出令牌$( \hat{t}_I^i, \hat{t}_c^i, \hat{t}_r^i)_{i=1}^N$。其中，注册令牌$\hat{t}_r^i$在后续预测中被丢弃。

**3. 预测头 (Prediction Heads)**
Transformer的输出令牌被送入不同的预测头，以推断各种3D属性。

*   **相机头 (Camera Head)**
    *   **输入**: 利用Transformer输出的相机令牌$(\hat{t}_c^i)_{i=1}^N$。
    *   **处理**: 经过四个额外的自注意力层和一个线性层。
    *   **输出**: 预测每张图像的相机参数$g_i \in R^9$（包括相机内参和外参）。这些参数定义在第一个相机的坐标系下，第一个相机的外参被设定为单位矩阵，即旋转四元数$q_1 =$，平移向量$t_1 =$。

*   **DPT 头 (Dense Predictions)**
    *   **输入**: 利用Transformer输出的图像令牌$(\hat{t}_I^i)_{i=1}^N$。
    *   **处理**: 图像令牌首先通过一个DPT (Dense Prediction Transformer) 层 转换为密集特征图$F_i \in R^{C'' \times H \times W}$。接着，每个$F_i$通过一个$3 \times 3$卷积层。
    *   **输出**:
        *   **深度图 (Depth Maps)**: $D_i \in R^{H \times W}$，表示从第$i$个相机观察到的每个像素对应的深度值。
        *   **点图 (Point Maps)**: $P_i \in R^{3 \times H \times W}$，将每个像素位置与其对应的3D场景点关联起来。与DUSt3R 类似，点图是视图不变的，意味着3D点$P_i(y)$定义在第一个相机$g_1$的坐标系下，作为世界参考坐标系。
        *   **跟踪特征 (Tracking Features)**: $T_i \in R^{C \times H \times W}$，这些密集特征作为点跟踪模块的输入。
    *   **不确定性预测**: DPT头还会预测深度图和点图的预测不确定性（aleatoric uncertainty）$\Sigma_D^i \in R^{H \times W}$和$\Sigma_P^i \in R^{H \times W}$。这些不确定性图在训练损失中被使用，训练后它们与模型的预测置信度成比例。

*   **跟踪模块 (Tracking Module)**
    *   **输入**: DPT头输出的密集跟踪特征$T_i$。
    *   **处理**: VGGT使用CoTracker2架构 作为跟踪模块。给定一个查询图像$I_q$中的查询点$y_q$，跟踪模块首先在查询图像的特征图$T_q$中对$y_q$进行双线性采样以获得其特征。然后，该特征与所有其他特征图$T_i$（$i \neq q$）相关联，生成一组关联图。这些关联图随后通过自注意力层处理，以预测最终的2D点$\hat{y}_{j,i}$。
    *   **输出**: 预测一组2D点轨迹$T^*(y_q) = (\hat{y}_i)_{i=1}^N$，表示$y_q$在所有图像$I_i$中对应的2D点。

**4. 训练 (Training)**
VGGT模型是端到端训练的，使用多任务损失函数，包括相机损失($\mathcal{L}_{camera}$)、深度损失($\mathcal{L}_{depth}$)、点图损失($\mathcal{L}_{pmap}$)和跟踪损失($\mathcal{L}_{track}$)。