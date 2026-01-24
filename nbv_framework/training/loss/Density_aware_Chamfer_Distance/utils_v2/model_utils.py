"""点云度量工具（密度感知倒角距离/倒角距离/地球移动距离）及近邻辅助函数。

该版本依赖本地度量封装，并使用批量向量化计算以提升速度。
本文件强调输出的含义和张量形状，便于与训练损失对齐。
"""

import math
import os
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .metrics import cd, emd, fscore


def calc_dcd(x, gt, alpha=40, n_lambda=0.5, return_raw=False, non_reg=False):
    """密度感知倒角距离。

    参数：
        x: 预测点云，形状 (B, N, 3)，B 为 batch。
        gt: 真值点云，形状 (B, M, 3)。
        alpha: 距离指数加权的温度系数，越大越强调近邻匹配。
        n_lambda: 频次惩罚的幂指数，越大越惩罚“多对一”的密集匹配。
        return_raw: 为 True 时追加倒角距离的中间量，便于可视化或调试。
        non_reg: 为 True 时将点数比例下限限制为 1，避免样本点数不平衡时
                 权重被过度缩小（更偏向稳定而非严格比例）。

    返回：
        列表结构固定，按顺序包含：
        1) loss: DCD 损失，形状 (B,)。
        2) cd_p: 倒角距离的开方版本，形状 (B,)。
        3) cd_t: 倒角距离的平方版本，形状 (B,)。
        若 return_raw=True，还会在末尾追加 4 个中间量：
        4) dist1: gt->x 的最近邻平方距离，形状 (B, M)。
        5) dist2: x->gt 的最近邻平方距离，形状 (B, N)。
        6) idx1: 与 dist1 对应的最近邻索引（指向 x），形状 (B, M)。
        7) idx2: 与 dist2 对应的最近邻索引（指向 gt），形状 (B, N)。
    """
    x = x.float()
    gt = gt.float()
    batch_size, n_x, _ = x.shape
    batch_size, n_gt, _ = gt.shape
    assert x.shape[0] == gt.shape[0]

    if non_reg:
        # 避免点数不平衡时权重被缩小（至少按 1 的比例放缩）。
        frac_12 = max(1, n_x / n_gt)
        frac_21 = max(1, n_gt / n_x)
    else:
        # 正常按点数比例做尺度调整。
        frac_12 = n_x / n_gt
        frac_21 = n_gt / n_x

    cd_p, cd_t, dist1, dist2, idx1, idx2 = calc_cd(x, gt, return_raw=True)
    # dist1 (B, M)：每个 gt 点在 x 中的最近邻平方距离。
    # idx1  (B, M)：对应最近邻在 x 中的索引。
    # dist2 (B, N) 与 idx2 (B, N)：反向对应（x 到 gt）。
    # 通过指数项强化近邻对应：距离越小，exp(-alpha * dist) 越接近 1。
    exp_dist1, exp_dist2 = torch.exp(-dist1 * alpha), torch.exp(-dist2 * alpha)

    # 统计每个预测点被多少 gt 点匹配（按 batch）。
    # count1 的形状为 (B, N)，其中 count1[b, i] 表示第 b 个样本中
    # 预测点 i 被多少 gt 点选为最近邻。
    count1 = torch.zeros_like(idx2)
    count1.scatter_add_(1, idx1.long(), torch.ones_like(idx1))
    # 使用匹配频次作为密度惩罚权重：被频繁匹配的点权重更小。
    # detach 保证频次统计不参与梯度传播，避免对匹配计数求导。
    weight1 = count1.gather(1, idx1.long()).float().detach() ** n_lambda
    weight1 = (weight1 + 1e-6) ** (-1) * frac_21
    # loss1 对应 gt->x 方向的密度感知损失（均值按 gt 点求）。
    loss1 = (1 - exp_dist1 * weight1).mean(dim=1)

    # 统计每个 gt 点被多少预测点匹配（按 batch）。
    # count2 的形状为 (B, M)，其中 count2[b, j] 表示第 b 个样本中
    # gt 点 j 被多少预测点选为最近邻。
    count2 = torch.zeros_like(idx1)
    count2.scatter_add_(1, idx2.long(), torch.ones_like(idx2))
    weight2 = count2.gather(1, idx2.long()).float().detach() ** n_lambda
    weight2 = (weight2 + 1e-6) ** (-1) * frac_12
    # loss2 对应 x->gt 方向的密度感知损失（均值按 x 点求）。
    loss2 = (1 - exp_dist2 * weight2).mean(dim=1)

    # 双向匹配的对称损失（两个方向取平均）。
    loss = (loss1 + loss2) / 2

    res = [loss, cd_p, cd_t]
    if return_raw:
        res.extend([dist1, dist2, idx1, idx2])

    return res

def calc_cd(output, gt, calc_f1=False, return_raw=False, normalize=False, separate=False):
    """倒角距离。

    参数：
        output: 预测点云，形状 (B, N, 3)。
        gt: 真值点云，形状 (B, M, 3)。
        calc_f1: 为 True 时追加 F 分数（基于 dist1/dist2 统计）。
        return_raw: 为 True 时追加原始 (dist1, dist2, idx1, idx2)。
        normalize: 预留参数，当前未使用（保持接口兼容）。
        separate: 为 True 时分别返回双向分量（方向性统计）。

    返回：
        默认返回 [cd_p, cd_t]：
        - cd_p：基于开方距离的倒角距离（更接近 L2）。
        - cd_t：基于平方距离的倒角距离（更接近 L2^2）。
        若 separate=True，则返回两个拼接张量（分别表示双向的均值）。
        若 calc_f1=True，则在末尾追加 f1。
        若 return_raw=True，则在末尾追加 dist1/dist2/idx1/idx2。
    """
    # 使用本地封装的倒角距离实现。
    cham_loss = cd()
    # dist1: gt->output 最近邻平方距离；dist2: output->gt 最近邻平方距离。
    dist1, dist2, idx1, idx2 = cham_loss(gt, output)
    # cd_p 使用开方距离，cd_t 使用平方距离。
    cd_p = (torch.sqrt(dist1).mean(1) + torch.sqrt(dist2).mean(1)) / 2
    cd_t = (dist1.mean(1) + dist2.mean(1))

    if separate:
        res = [torch.cat([torch.sqrt(dist1).mean(1).unsqueeze(0), torch.sqrt(dist2).mean(1).unsqueeze(0)]),
               torch.cat([dist1.mean(1).unsqueeze(0),dist2.mean(1).unsqueeze(0)])]
    else:
        res = [cd_p, cd_t]
    if calc_f1:
        # fscore 返回 (f1, 精确率, 召回率)。
        f1, _, _ = fscore(dist1, dist2)
        res.append(f1)
    if return_raw:
        res.extend([dist1, dist2, idx1, idx2])
    return res

def calc_emd(output, gt, eps=0.005, iterations=50):
    """点云之间的地球移动距离。"""
    # 使用本地封装的地球移动距离实现。
    # eps 与 iterations 控制数值精度与迭代次数。
    emd_loss = emd()
    dist, _ = emd_loss(output, gt, eps, iterations)
    # 地球移动距离返回每点的平方距离。
    emd_out = torch.sqrt(dist).mean(1)
    return emd_out

def knn(x, k):
    """计算点云中每个点的 k 近邻索引。

    参数：
        x: 点特征，形状 (B, C, N)，C 为通道数，N 为点数。
        k: 近邻数量。

    返回：
        idx: 形状 (B, N, k) 的索引张量，每个点返回 k 个邻居的索引。
    """
    # 成对负平方距离（数值越大越近）。
    inner = -2 * torch.matmul(x.transpose(2, 1).contiguous(), x)
    xx = torch.sum(x ** 2, dim=1, keepdim=True)
    pairwise_distance = -xx - inner - xx.transpose(2, 1).contiguous()
    idx = pairwise_distance.topk(k=k, dim=-1)[1]
    return idx

def knn_point(pk, point_input, point_output):
    """从输出点到输入点的 k 近邻查询。

    参数：
        pk: 每个输出点的近邻数量。
        point_input: 输入点，形状 (B, N, C)。
        point_output: 输出点，形状 (B, M, C)。

    返回：
        dist: 负平方距离，形状 (B, M, pk)，值越大表示越近。
        idx: 在 point_input 中的索引，形状 (B, M, pk)。
    """
    m = point_output.size()[1]
    n = point_input.size()[1]

    # 计算输出点与输入点的成对距离，得到 (B, M, N) 的距离矩阵。
    inner = -2 * torch.matmul(point_output, point_input.transpose(2, 1).contiguous())
    xx = torch.sum(point_output ** 2, dim=2, keepdim=True).repeat(1, 1, n)
    yy = torch.sum(point_input ** 2, dim=2, keepdim=False).unsqueeze(1).repeat(1, m, 1)
    pairwise_distance = -xx - inner - yy
    dist, idx = pairwise_distance.topk(k=pk, dim=-1)
    return dist, idx

def knn_point_all(pk, point_input, point_output):
    """与 knn_point 相同，保留用于接口兼容。

    该函数与 knn_point 的实现完全一致，仅用于保持旧代码的调用方式。
    """
    m = point_output.size()[1]
    n = point_input.size()[1]

    inner = -2 * torch.matmul(point_output, point_input.transpose(2, 1).contiguous())
    xx = torch.sum(point_output ** 2, dim=2, keepdim=True).repeat(1, 1, n)
    yy = torch.sum(point_input ** 2, dim=2, keepdim=False).unsqueeze(1).repeat(1, m, 1)
    pairwise_distance = -xx - inner - yy
    dist, idx = pairwise_distance.topk(k=pk, dim=-1)

    return dist, idx

if __name__ == "__main__":
    # 简单的显卡张量自检。
    pc1 = torch.randn([1,5,3]).cuda()
    pc2 = torch.randn([1,5,3]).cuda()
    dcd, _, _ = calc_dcd(pc1, pc2)
    print(dcd)
