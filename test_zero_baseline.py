# -*- coding: utf-8 -*-
import os  # 操作系统路径等相关操作
import argparse  # 命令行参数解析库
import random  # Python 自带随机数库
import math  # 数学函数库
import numpy as np  # 数值计算库
import torch  # PyTorch 主库
from torch import nn  # 神经网络模块
from torch.nn import functional as F  # 常用函数接口（卷积、插值等）
from tqdm import tqdm  # 进度条显示
from sklearn.metrics import roc_auc_score  # ROC-AUC 评价指标
from scipy.ndimage import gaussian_filter  # 高斯滤波（本文件未使用）
from dataset.medical_zero import MedTestDataset, MedTrainDataset  # 零样本训练/测试数据集定义
from CLIP.clip import create_model  # 创建 CLIP 模型
from CLIP.tokenizer import tokenize  # CLIP 文本 tokenizer（本文件未直接使用）
from CLIP.adapter import CLIP_Inplanted  # 在 CLIP 中插入适配器的封装模型
from PIL import Image  # 图像处理（本文件未直接使用）
from sklearn.metrics import precision_recall_curve  # PR 曲线（本文件未使用）
from loss import FocalLoss, BinaryDiceLoss  # 自定义 Focal loss 和 Dice loss
from utils import augment, encode_text_with_prompt_ensemble, encode_text_with_hyperbolic_adjustment  # 数据增强和文本编码工具
from prompt import REAL_NAME  # 各任务真实名称字典，用于文本 prompt
import geoopt  # 双曲几何库（用于 Hyper-MVFA）

import warnings  # 警告过滤
warnings.filterwarnings("ignore")  # 忽略所有警告，避免日志过多

# 检测是否有可用 GPU
use_cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if use_cuda else "cpu")  # 有 GPU 则用 cuda:0，否则用 CPU

# 各数据集任务到整数索引的映射（>0 表示有像素级标注，<=0 表示只做图像级）
CLASS_INDEX = {'Brain':3, 'Liver':2, 'Retina_RESC':1, 'Retina_OCT2017':-1, 'Chest':-2, 'Histopathology':-3}
# 反向映射：由索引恢复任务名称
CLASS_INDEX_INV = {3:'Brain', 2:'Liver', 1:'Retina_RESC', -1:'Retina_OCT2017', -2:'Chest', -3:'Histopathology'}


def setup_seed(seed):
    """
    设置随机种子，保证实验可复现
    """
    torch.manual_seed(seed)  # CPU 随机数种子
    torch.cuda.manual_seed_all(seed)  # 所有 GPU 随机数种子
    np.random.seed(seed)  # numpy 随机数种子
    random.seed(seed)  # Python 内置随机数种子
    torch.backends.cudnn.deterministic = True  # cuDNN 使用确定性算法
    torch.backends.cudnn.benchmark = False  # 关闭 benchmark，避免非确定性行为


def main():
    """
    零样本测试脚本主入口：
    1）加载预训练好的 zero-shot 适配器权重
    2）在指定数据集上进行图像级 / 像素级异常检测评估
    """
    parser = argparse.ArgumentParser(description='Testing')  # 命令行参数解析器
    parser.add_argument('--model_name', type=str, default='ViT-L-14-336', help="ViT-B-16-plus-240, ViT-L-14-336")  # CLIP backbone 名称
    parser.add_argument('--pretrain', type=str, default='openai', help="laion400m, openai")  # 预训练权重来源
    parser.add_argument('--obj', type=str, default='Retina_RESC')  # 当前测试的数据集/任务
    parser.add_argument('--data_path', type=str, default='./data/')  # 数据路径
    parser.add_argument('--batch_size', type=int, default=1)  # batch 大小
    parser.add_argument('--img_size', type=int, default=240)  # 输入图像尺寸
    parser.add_argument('--save_path', type=str, default=None, help='checkpoint dir')  # 预训练 zero-shot checkpoint 保存路径
    parser.add_argument("--epoch", type=int, default=50, help="epochs")  # 该脚本中未使用的训练轮数参数（保留一致性）
    parser.add_argument("--learning_rate", type=float, default=0.0001, help="learning rate")  # 用于定义优化器的学习率（本测试脚本中基本不使用优化）
    parser.add_argument("--features_list", type=int, nargs="+", default=[6, 12, 18, 24], help="features used")  # 使用 CLIP 的哪些层作为特征
    parser.add_argument('--seed', type=int, default=111)  # 随机种子
    # Hyper-MVFA 双曲模式参数
    parser.add_argument('--use_hyperbolic', action='store_true', help='Use hyperbolic adapters and distances')
    parser.add_argument('--hyperbolic_c', type=float, default=0.1, help='Curvature of Poincare ball')
    parser.add_argument('--scale_normal', type=float, default=0.1, help='Radius scale for normal text embeddings')
    parser.add_argument('--scale_abnormal', type=float, default=0.8, help='Radius scale for abnormal text embeddings')
    parser.add_argument('--temperature', type=float, default=20.0, help='Temperature for scaling hyperbolic distances to logits')
    args = parser.parse_args()  # 解析命令行参数

    # ===== 自动根据是否使用双曲模式选择默认读取目录 =====
    if args.save_path is None:
        mode_tag = "zero-shot-hyper" if args.use_hyperbolic else "zero-shot-euclid"
        args.save_path = os.path.join("./ckpt", mode_tag)

    # 设置随机种子
    setup_seed(args.seed)
    
    # 固定特征提取器：创建预训练 CLIP 模型
    clip_model = create_model(model_name=args.model_name,  # backbone 名称
                              img_size=args.img_size,      # 输入图片大小
                              device=device,               # 设备
                              pretrained=args.pretrain,    # 预训练权重来源
                              require_pretrained=True)     # 强制需要预训练权重
    clip_model.eval()  # CLIP 模型设为 eval 模式（不训练 CLIP 本体）

    # 在 CLIP 基础上插入适配器，构建 MVFA 模型
    model = CLIP_Inplanted(clip_model=clip_model, 
                           features=args.features_list,
                           use_hyperbolic=args.use_hyperbolic,
                           hyperbolic_c=args.hyperbolic_c).to(device)
    model.eval()  # 设为 eval 模式

    # 加载 previously 训练好的 zero-shot 适配器参数
    checkpoint = torch.load(os.path.join(f'{args.save_path}', f'{args.obj}.pth'))  # 从 save_path/obj.pth 中读取
    model.seg_adapters.load_state_dict(checkpoint["seg_adapters"])  # 加载分割适配器权重
    model.det_adapters.load_state_dict(checkpoint["det_adapters"])  # 加载检测适配器权重

    # 为适配器构建优化器（此测试脚本中并不会实际训练，只是保留与 train_zero 同样的结构）
    seg_optimizer = torch.optim.Adam(list(model.seg_adapters.parameters()), lr=args.learning_rate, betas=(0.5, 0.999))
    det_optimizer = torch.optim.Adam(list(model.det_adapters.parameters()), lr=args.learning_rate, betas=(0.5, 0.999))


    # 加载数据集和 DataLoader
    kwargs = {'num_workers': 0, 'pin_memory': True} if use_cuda else {}  # 设置为 0 避免 /tmp 空间不足问题
    train_dataset = MedTrainDataset(args.data_path, args.obj, args.img_size, args.batch_size)  # 训练集（此脚本中未实际使用）
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=1, shuffle=True, **kwargs)

    test_dataset = MedTestDataset(args.data_path, args.obj, args.img_size)  # 测试集
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=False, **kwargs)  # 测试 DataLoader


    # 定义损失函数（此脚本只评估，不反向，但保持与训练脚本一致）
    loss_focal = FocalLoss()  # 像素级 Focal Loss（未使用）
    loss_dice = BinaryDiceLoss()  # 像素级 Dice Loss（未使用）
    loss_bce = torch.nn.BCEWithLogitsLoss()  # 图像级二分类 BCE Loss（未使用）


    # 文本特征列表，0 位置占位，不使用
    text_feature_list = [0]
    ball_list = [None]  # 保存每个任务对应的 ball 对象（双曲模式时使用）
    
    # 文本 prompt 编码：为各个数据集任务预先计算文本特征（[768, 2] 等维度）
    with torch.cuda.amp.autocast(), torch.no_grad():  # 混合精度 + 不需要梯度
        for i in [1,2,3,-3,-2,-1]:  # 对 CLASS_INDEX 中出现的索引进行遍历
            # 通过索引反查任务名称，再从 REAL_NAME 中取真实名称列表，做 prompt ensemble 编码
            if args.use_hyperbolic:
                # 使用双曲模式：调用双曲文本编码 + 半径调整
                text_feature, ball = encode_text_with_hyperbolic_adjustment(
                    clip_model,
                    REAL_NAME[CLASS_INDEX_INV[i]],
                    device,
                    use_hyperbolic=True,
                    c=args.hyperbolic_c,
                    scale_normal=args.scale_normal,
                    scale_abnormal=args.scale_abnormal
                )
                ball_list.append(ball)
            else:
                # 使用欧氏模式：调用原始文本编码
                text_feature = encode_text_with_prompt_ensemble(clip_model, REAL_NAME[CLASS_INDEX_INV[i]], device)
                ball_list.append(None)
            
            text_feature_list.append(text_feature)  # 追加到列表中

    # 使用当前任务对应的文本特征在测试集上做评估
    score = test(args, model, test_loader, text_feature_list[CLASS_INDEX[args.obj]], ball_list[CLASS_INDEX[args.obj]], args.temperature)
        


def test(args, seg_model, test_loader, text_features, ball=None, temperature=20.0):
    """
    在测试集上评估 zero-shot 模型：
    - image_scores：图像级异常得分
    - segment_scores：像素级异常得分（若有分割任务）
    返回：像素级 AUC + 图像级 AUC 或仅图像级 AUC
    
    Args:
        args: 命令行参数
        seg_model: 模型
        test_loader: 测试数据加载器
        text_features: 文本特征 [C, 2]
        ball: PoincareBall 对象（双曲模式时使用，欧氏模式为 None）
    """
    gt_list = []  # 图像级 GT 标签列表
    gt_mask_list = []  # 像素级 GT mask 列表
    image_scores = []  # 图像级预测分数列表
    segment_scores = []  # 像素级预测得分图列表
    
    # 遍历测试集
    for (image, y, mask) in tqdm(test_loader):
        image = image.to(device)  # 将图像送至 GPU/CPU
        # 将 mask 二值化：>0.5 为 1，其余为 0
        mask[mask > 0.5], mask[mask <= 0.5] = 1, 0

        # 测试阶段不计算梯度，启用混合精度
        with torch.no_grad(), torch.cuda.amp.autocast():
            # 前向传播：得到分割与检测的 patch tokens
            _, ori_seg_patch_tokens, ori_det_patch_tokens = seg_model(image)
            
            # 遍历 batch 中的每个样本
            batch_size_current = image.shape[0]
            for batch_idx in range(batch_size_current):
                # 提取当前样本的 tokens，去掉 CLS token（索引 0）
                ori_seg_patch_tokens_single = [p[batch_idx, 1:, :] for p in ori_seg_patch_tokens]
                ori_det_patch_tokens_single = [p[batch_idx, 1:, :] for p in ori_det_patch_tokens]
                
                # ------------------ 图像级分数（image-level detection）------------------
                anomaly_score = 0  # 初始化当前图像的总异常得分
                patch_tokens = ori_det_patch_tokens_single.copy()  # 复制一份检测 patch token 列表
                for layer in range(len(patch_tokens)):
                    if args.use_hyperbolic:
                        # ===== 双曲模式 =====
                        L, C = patch_tokens[layer].shape
                        
                        # 文本特征：[C, 2] -> [2, C]
                        text_h = text_features.T  # [2, C]
                        normal_text = text_h[0]  # [C]
                        abnormal_text = text_h[1]  # [C]
                        
                        # 向量化计算距离
                        dist_normal = ball.dist(patch_tokens[layer], normal_text)  # [L]
                        dist_abnormal = ball.dist(patch_tokens[layer], abnormal_text)  # [L]
                        
                        # 距离转 logits
                        logits_normal = -temperature * dist_normal
                        logits_abnormal = -temperature * dist_abnormal
                        
                        # Stack 成 [L, 2]
                        anomaly_map = torch.stack([logits_normal, logits_abnormal], dim=-1).unsqueeze(0)  # [1, L, 2]
                        # 对类别做 softmax，取异常类（索引 1）的概率
                        anomaly_map = torch.softmax(anomaly_map, dim=-1)[:, :, 1]
                        # 对所有 patch 求均值，累加至 anomaly_score
                        anomaly_score += anomaly_map.mean()
                    else:
                        # ===== 欧氏模式 =====
                        # 对每个 patch 特征做 L2 归一化
                        patch_tokens[layer] /= patch_tokens[layer].norm(dim=-1, keepdim=True)
                        # 与文本特征相乘，得到 [L, 2] logits，乘以 100.0 作为放缩
                        anomaly_map = (100.0 * patch_tokens[layer] @ text_features).unsqueeze(0)
                        # 对类别维 softmax，取异常类（索引 1）的概率
                        anomaly_map = torch.softmax(anomaly_map, dim=-1)[:, :, 1]
                        # 对所有 patch 求均值，累加至 anomaly_score
                        anomaly_score += anomaly_map.mean()
                # 将 anomaly_score 移到 CPU 并保存
                image_scores.append(anomaly_score.cpu())

                # ------------------ 像素级分数（pixel-level segmentation）------------------
                patch_tokens = ori_seg_patch_tokens_single  # 使用分割 head 的 patch token
                anomaly_maps = []  # 存放各层的 anomaly map
                for layer in range(len(patch_tokens)):
                    if args.use_hyperbolic:
                        # ===== 双曲模式 =====
                        L, C = patch_tokens[layer].shape
                        H = int(np.sqrt(L))  # patch 网格尺寸 H x H
                        
                        # 文本特征：[C, 2] -> [2, C]
                        text_h = text_features.T  # [2, C]
                        normal_text = text_h[0]  # [C]
                        abnormal_text = text_h[1]  # [C]
                        
                        # 向量化计算距离
                        dist_normal = ball.dist(patch_tokens[layer], normal_text)  # [L]
                        dist_abnormal = ball.dist(patch_tokens[layer], abnormal_text)  # [L]
                        
                        # 距离转 logits
                        logits_normal = -temperature * dist_normal
                        logits_abnormal = -temperature * dist_abnormal
                        
                        # Stack 成 [L, 2]
                        anomaly_map = torch.stack([logits_normal, logits_abnormal], dim=-1).unsqueeze(0)  # [1, L, 2]
                        B = 1
                        # 将 [1, L, 2] 变形并插值到 img_size×img_size
                        anomaly_map = F.interpolate(anomaly_map.permute(0, 2, 1).view(B, 2, H, H),
                                                    size=args.img_size, mode='bilinear', align_corners=True)
                        # 对类别维 softmax 后取异常通道（索引 1）的概率 map
                        anomaly_map = torch.softmax(anomaly_map, dim=1)[:, 1, :, :]
                        # 转成 numpy 存入列表
                        anomaly_maps.append(anomaly_map.cpu().numpy())
                    else:
                        # ===== 欧氏模式 =====
                        # L2 归一化
                        patch_tokens[layer] /= patch_tokens[layer].norm(dim=-1, keepdim=True)
                        # 与文本特征相乘，得到 [L, 2] logits
                        anomaly_map = (100.0 * patch_tokens[layer] @ text_features).unsqueeze(0)
                        B, L, C = anomaly_map.shape  # B: batch, L: patch 数量, C: 类别数
                        H = int(np.sqrt(L))  # 假设 patch 数量为 H*H，因此 H = sqrt(L)
                        # 将 [B, L, C] 调整为 [B, C, H, H]，再插值到 img_size x img_size
                        anomaly_map = F.interpolate(anomaly_map.permute(0, 2, 1).view(B, 2, H, H),
                                                    size=args.img_size, mode='bilinear', align_corners=True)
                        # 对类别维 softmax 后取异常通道（索引 1）的概率 map
                        anomaly_map = torch.softmax(anomaly_map, dim=1)[:, 1, :, :]
                        # 转成 numpy 存入列表
                        anomaly_maps.append(anomaly_map.cpu().numpy())
                # 对所有层的 anomaly map 求和，得到最终像素级得分图
                final_score_map = np.sum(anomaly_maps, axis=0)
                
                # 收集当前样本的 GT mask、GT 标签和预测得分图
                gt_mask_list.append(mask[batch_idx].squeeze().cpu().detach().numpy())
                gt_list.extend(y[batch_idx:batch_idx+1].cpu().detach().numpy())
                segment_scores.append(final_score_map)
        
        

    # 将列表转换为 numpy 数组
    gt_list = np.array(gt_list)  # 图像级 GT
    gt_mask_list = np.asarray(gt_mask_list)  # 像素级 GT
    gt_mask_list = (gt_mask_list>0).astype(np.int_)  # 再次确保为 0/1

    segment_scores = np.array(segment_scores)  # 像素级预测得分图
    image_scores = np.array(image_scores)  # 图像级预测分数

    # 对像素级和图像级得分分别做 min-max 归一化到 [0, 1]
    segment_scores = (segment_scores - segment_scores.min()) / (segment_scores.max() - segment_scores.min())
    image_scores = (image_scores - image_scores.min()) / (image_scores.max() - image_scores.min())

    # 计算图像级 ROC AUC
    img_roc_auc_det = roc_auc_score(gt_list, image_scores)
    print(f'{args.obj} AUC : {round(img_roc_auc_det,4)}')

    # 若该任务具有像素级标注（即 CLASS_INDEX[obj] > 0），则同时计算像素级 ROC AUC
    if CLASS_INDEX[args.obj] > 0:
        seg_roc_auc = roc_auc_score(gt_mask_list.flatten(), segment_scores.flatten())
        print(f'{args.obj} pAUC : {round(seg_roc_auc,4)}')
        # 返回像素级 AUC + 图像级 AUC 的和，作为综合指标
        return seg_roc_auc + img_roc_auc_det
    else:
        # 否则只返回图像级 AUC
        return img_roc_auc_det

# 当该文件作为脚本直接运行时，执行 main()
if __name__ == '__main__':
    main()
