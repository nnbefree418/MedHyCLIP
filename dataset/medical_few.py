import os  # 文件路径相关操作
import torch  # PyTorch 主库
from torch.utils.data import Dataset  # 数据集基类
from torchvision import transforms  # 常用图像预处理
from PIL import Image  # 图像读写库
import random  # 随机选择 few-shot 样本
import pandas as pd  # 本文件未使用
import numpy as np  # 数值计算库

# 所有支持的数据集名称列表
CLASS_NAMES = ['Brain', 'Liver', 'Retina_RESC', 'Retina_OCT2017', 'Chest', 'Histopathology']

# 数据集名称到任务索引的映射（>0 表示存在像素级 mask；≤0 表示只有图像级标签）
CLASS_INDEX = {'Brain':3, 'Liver':2, 'Retina_RESC':1, 'Retina_OCT2017':-1, 'Chest':-2, 'Histopathology':-3}


class MedDataset(Dataset):
    """
    Few-shot 医学异常检测数据集（用于 few-shot 测试阶段）
    本类用于加载：
        - 测试集图像
        - few-shot 支持集（正常样本/异常样本）
    """

    def __init__(self,
                 dataset_path='/data/',      # 数据根路径
                 class_name='Brain',         # 数据集名称
                 resize=240,                 # 图像 resize 大小
                 shot=4,                     # few-shot 样本数量
                 iterate=-1                  # 若 >=0 则从固定文件读取指定 few-shot 样本
                 ):
        # 检查输入的数据集名称是否有效
        assert class_name in CLASS_NAMES, 'class_name: {}, should be in {}'.format(class_name, CLASS_NAMES)
        # few-shot 样本数必须 > 0
        assert shot > 0, 'shot number : {}, should be positive integer'.format(shot)

        # 完整数据集路径，如 ./data/Brain_AD/
        self.dataset_path = os.path.join(dataset_path, f'{class_name}_AD')
        self.resize = resize  # 图像 resize 尺寸
        self.shot = shot  # few-shot 样本数量
        self.iterate = iterate  # 是否从 fewshot_seed 中加载固定分割
        self.class_name = class_name  # 数据集名称
        self.seg_flag = CLASS_INDEX[class_name]  # 若 >0 则该任务有像素级标注

        # -----------------------------
        # 加载测试集所有图像路径、标签、mask 路径
        # -----------------------------
        self.x, self.y, self.mask = self.load_dataset_folder(self.seg_flag)

        # -----------------------------
        # 定义图像与 Mask 的 transform
        # -----------------------------
        self.transform_x = transforms.Compose([
            transforms.Resize((resize, resize), Image.BICUBIC),  # 双三次插值 resize
            transforms.ToTensor(),  # 转为 PyTorch 张量
        ])

        self.transform_mask = transforms.Compose([
            transforms.Resize((resize, resize), Image.NEAREST),  # mask 使用最近邻插值
            transforms.ToTensor()
        ])

        # -----------------------------
        # 获取 few-shot 支持样本
        # -----------------------------
        self.fewshot_norm_img = self.get_few_normal()  # few-shot 正常样本
        self.fewshot_abnorm_img, self.fewshot_abnorm_mask = self.get_few_abnormal()  # few-shot 异常样本（含 mask）
        

    def __getitem__(self, idx):
        """
        返回测试集中的一条样本，包括：
            x_img: 图像张量
            y: 图像级标签 0/1
            mask: 若 seg_flag > 0 则有 mask，否则返回全 0 的 mask
        """
        x, y, mask = self.x[idx], self.y[idx], self.mask[idx]

        # 读取 RGB 图像
        x = Image.open(x).convert('RGB')
        x_img = self.transform_x(x)

        # 若没有像素级标注，mask 返回全 0
        if self.seg_flag < 0:
            return x_img, y, torch.zeros([1, self.resize, self.resize])

        # 若该图像没有 mask，则表示正常图像
        if mask is None:
            mask = torch.zeros([1, self.resize, self.resize])
            y = 0  # 正常标签
        else:
            # 有 mask，则读取异常 mask
            mask = Image.open(mask).convert('L')
            mask = self.transform_mask(mask)
            y = 1  # 异常标签

        return x_img, y, mask

    def __len__(self):
        """ 数据集大小 """
        return len(self.x)

    def load_dataset_folder(self, seg_flag):
        """
        加载测试集数据
        路径格式：
            dataset/test/good/img/*.png
            dataset/test/Ungood/img/*.png
            dataset/test/Ungood/anomaly_mask/*.png  (仅 seg_flag>0 时存在)
        """
        x, y, mask = [], [], []

        # ------------------- 加载正常样本 -------------------
        normal_img_dir = os.path.join(self.dataset_path, 'test', 'good', 'img')
        img_fpath_list = sorted([os.path.join(normal_img_dir, f) for f in os.listdir(normal_img_dir)])
        x.extend(img_fpath_list)
        y.extend([0] * len(img_fpath_list))  # 正常类别为 0
        mask.extend([None] * len(img_fpath_list))  # 正常样本无 mask

        # ------------------- 加载异常样本 -------------------
        abnormal_img_dir = os.path.join(self.dataset_path, 'test', 'Ungood', 'img')
        img_fpath_list = sorted([os.path.join(abnormal_img_dir, f) for f in os.listdir(abnormal_img_dir)])
        x.extend(img_fpath_list)
        y.extend([1] * len(img_fpath_list))  # 异常类别为 1

        # 若该任务有像素级标注，则对应 mask 路径
        if self.seg_flag > 0:
            gt_fpath_list = [f.replace('img', 'anomaly_mask') for f in img_fpath_list]
            mask.extend(gt_fpath_list)
        else:
            mask.extend([None] * len(img_fpath_list))

        # 数据一致性检查
        assert len(x) == len(y), 'number of x and y should be same'
        return list(x), list(y), list(mask)


    def get_few_normal(self):
        """
        获取 few-shot 正常支持样本（good / valid）
        若 iterate >= 0，则按指定 seed 文件选样本
        """
        x = []
        img_dir = os.path.join(self.dataset_path, 'valid', 'good', 'img')
        normal_names = os.listdir(img_dir)

        # ------------------- 选择 few-shot 正常样本 -------------------
        if self.iterate < 0:
            # 随机采样 few-shot 图像名
            random_choice = random.sample(normal_names, self.shot)
        else:
            # 从 fewshot_seed 文件读取固定样本
            random_choice = []
            with open(f'./dataset/fewshot_seed/{self.class_name}/{self.shot}-shot.txt', 'r', encoding='utf-8') as infile:
                for line in infile:
                    data_line = line.strip("\n").split()
                    if data_line[0] == f'n-{self.iterate}:':
                        random_choice = data_line[1:]
                        break

        # 筛选文件，仅保留 png/jpeg
        for f in random_choice:
            if f.endswith('.png') or f.endswith('.jpeg'):
                x.append(os.path.join(img_dir, f))

        fewshot_img = []
        # ------------------- 读取 few-shot 图像 -------------------
        for idx in range(self.shot):
            image = x[idx]
            image = Image.open(image).convert('RGB')
            image = self.transform_x(image)
            fewshot_img.append(image.unsqueeze(0))  # 增加 batch 维度

        fewshot_img = torch.cat(fewshot_img)
        return fewshot_img


    def get_few_abnormal(self):
        """
        获取 few-shot 异常支持样本（Ungood / valid），含 mask（若该任务有像素标注）
        """
        x = []
        y = []
        img_dir = os.path.join(self.dataset_path, 'valid', 'Ungood', 'img')
        mask_dir = os.path.join(self.dataset_path, 'valid', 'Ungood', 'anomaly_mask')

        abnormal_names = os.listdir(img_dir)

        # ------------------- 选择 few-shot 异常样本 -------------------
        if self.iterate < 0:
            random_choice = random.sample(abnormal_names, self.shot)
        else:
            random_choice = []
            with open(f'./dataset/fewshot_seed/{self.class_name}/{self.shot}-shot.txt', 'r', encoding='utf-8') as infile:
                for line in infile:
                    data_line = line.strip("\n").split()
                    if data_line[0] == f'a-{self.iterate}:':  # 读取异常 few-shot seed
                        random_choice = data_line[1:]
                        break

        # 保留异常图像与其 mask 路径
        for f in random_choice:
            if f.endswith('.png') or f.endswith('.jpeg'):
                x.append(os.path.join(img_dir, f))          # 异常图像
                y.append(os.path.join(mask_dir, f))         # 对应 mask

        fewshot_img = []
        fewshot_mask = []

        # ------------------- 加载 few-shot 异常图像及 mask -------------------
        for idx in range(self.shot):
            image = x[idx]
            image = Image.open(image).convert('RGB')
            image = self.transform_x(image)
            fewshot_img.append(image.unsqueeze(0))

            # 若该任务有像素级标注，则加载对应 mask
            if CLASS_INDEX[self.class_name] > 0:
                image = y[idx]
                image = Image.open(image).convert('L')
                image = self.transform_mask(image)
                fewshot_mask.append(image.unsqueeze(0))

        fewshot_img = torch.cat(fewshot_img)

        # 若该任务没有 mask，则返回 None
        if len(fewshot_mask) == 0:
            return fewshot_img, None
        else:
            fewshot_mask = torch.cat(fewshot_mask)
            return fewshot_img, fewshot_mask
