import numpy as np  # 数值计算库
import torch  # PyTorch 主库
import torch.nn as nn  # 神经网络模块
import torch.nn.functional as F  # 常用函数模块
from math import exp  # 指数函数（本文件未直接使用）

class FocalLoss(nn.Module):
    """
    Focal Loss 的 PyTorch 实现版本
    参考来源: https://github.com/Hsuxu/Loss_ToolBox-PyTorch/blob/master/FocalLoss/FocalLoss.py

    Focal Loss 用于解决前景/背景严重不平衡的问题，
    公式：Focal_Loss = -α * (1 - pt)^γ * log(pt)
    其中：
        pt = 模型对真实类别的预测概率
        α  = 类别权重
        γ  = 控制难易样本的损失放大力度（γ 越大，越强调难分类样本）

    参数说明：
    :param apply_nonlin: 对 logit 输入进行非线性处理（如 sigmoid/softmax）
    :param alpha: 类别权重，可为 float / list / ndarray
    :param gamma: γ > 0 减少易分类样本的损失
    :param balance_index: 当 alpha 为 float 时，用于指定前景类别索引
    :param smooth: label smoothing 的平滑因子
    :param size_average: 若为 True，则求平均 loss
    """

    def __init__(self, apply_nonlin=None, alpha=None, gamma=2, balance_index=0, smooth=1e-5, size_average=True):
        super(FocalLoss, self).__init__()
        self.apply_nonlin = apply_nonlin  # 对输出 logits 进行非线性变换（如 softmax）
        self.alpha = alpha  # 类别权重
        self.gamma = gamma  # γ 参数
        self.balance_index = balance_index  # 平衡类别索引
        self.smooth = smooth  # 标签平滑因子
        self.size_average = size_average  # 是否对 batch 求平均

        # smooth 必须在 0-1 范围内
        if self.smooth is not None:
            if self.smooth < 0 or self.smooth > 1.0:
                raise ValueError('smooth value should be in [0,1]')

    def forward(self, logit, target):
        # 若指定了非线性激活，对输入 logit 做处理
        if self.apply_nonlin is not None:
            logit = self.apply_nonlin(logit)
        num_class = logit.shape[1]  # 类别数 C

        # 若 logit 维度大于 2，则说明是像素级预测（N,C,H,W），需要拉平成 (N*H*W, C)
        if logit.dim() > 2:
            # N,C,H,W → N,C,H*W
            logit = logit.view(logit.size(0), logit.size(1), -1)
            # N,C,M → N,M,C
            logit = logit.permute(0, 2, 1).contiguous()
            # → (N*M, C)
            logit = logit.view(-1, logit.size(-1))

        target = torch.squeeze(target, 1)  # 去掉通道维度
        target = target.view(-1, 1)  # 展平成列向量
        alpha = self.alpha  # 类别权重 α

        # α 参数的不同输入方式处理
        if alpha is None:
            alpha = torch.ones(num_class, 1)  # 默认每类权重相同
        elif isinstance(alpha, (list, np.ndarray)):
            # 若 α 是列表或 numpy 数组，需转为 Tensor 并归一化
            assert len(alpha) == num_class
            alpha = torch.FloatTensor(alpha).view(num_class, 1)
            alpha = alpha / alpha.sum()
        elif isinstance(alpha, float):
            # 若 α 是 float，则按 balance_index 指定前景类别权重
            alpha = torch.ones(num_class, 1)
            alpha = alpha * (1 - self.alpha)
            alpha[self.balance_index] = self.alpha
        else:
            raise TypeError('Not support alpha type')

        # 将 α 放到 logit 同一设备
        if alpha.device != logit.device:
            alpha = alpha.to(logit.device)

        idx = target.cpu().long()  # 将 target 变为 CPU long 类型，用于 scatter

        # 构建 one-hot 标签矩阵
        one_hot_key = torch.FloatTensor(target.size(0), num_class).zero_()
        one_hot_key = one_hot_key.scatter_(1, idx, 1)
        if one_hot_key.device != logit.device:
            one_hot_key = one_hot_key.to(logit.device)

        # 若使用 label smoothing
        if self.smooth:
            one_hot_key = torch.clamp(one_hot_key, self.smooth / (num_class - 1), 1.0 - self.smooth)

        # pt = 模型对真实类别的预测概率
        pt = (one_hot_key * logit).sum(1) + self.smooth
        logpt = pt.log()  # log(pt)

        gamma = self.gamma  # γ 参数

        # 根据标签选择 α 权重
        alpha = alpha[idx]
        alpha = torch.squeeze(alpha)

        # Focal loss 公式
        loss = -1 * alpha * torch.pow((1 - pt), gamma) * logpt

        # 若设定 size_average，则对 batch 求均值
        if self.size_average:
            loss = loss.mean()
        return loss


class BinaryDiceLoss(nn.Module):
    """
    Dice Loss 的二分类版本实现
    Dice 系数用于衡量预测区域与真实区域的重叠程度：
    
        Dice = (2 * |X ∩ Y| + smooth) / (|X| + |Y| + smooth)

    Dice Loss = 1 - Dice

    用于像素级分割任务，尤其适用于前景/背景极度不平衡的医学图像分割
    """

    def __init__(self):
        super(BinaryDiceLoss, self).__init__()

    def forward(self, input, targets):
        N = targets.size()[0]  # batch 大小
        smooth = 1  # 平滑项，避免分母为 0

        # 展平成 (N, H*W)
        input_flat = input.view(N, -1)  # 预测概率
        targets_flat = targets.view(N, -1)  # GT mask

        # 求交集：预测值 * 真实标签
        intersection = input_flat * targets_flat

        # Dice 系数
        N_dice_eff = (2 * intersection.sum(1) + smooth) / (input_flat.sum(1) + targets_flat.sum(1) + smooth)

        # Dice Loss = 1 - Dice
        loss = 1 - N_dice_eff.sum() / N
        return loss
