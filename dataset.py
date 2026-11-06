import os
import random
import numpy as np
from PIL import Image, ImageDraw
import torch
from torch.utils.data import Dataset

class SegDataset(Dataset):

    def __init__(self, image_dir, mask_dir, image_size=128, augment=False):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.image_size = image_size
        self.augment = augment

        # 找出所有图片文件，排序保证每次顺序一致
        names = sorted(os.listdir(image_dir))
        self.names = [n for n in names if n.lower().endswith((".png", ".jpg", ".jpeg", ".bmp", ".tif"))]
        if len(self.names) == 0:
            raise RuntimeError("文件夹里没有图片：" + image_dir)

    def __len__(self):
        return len(self.names)

    def __getitem__(self, index):
        name = self.names[index]

        # 1. 读图：原图转成 RGB（3 通道），掩码转成灰度（1 通道）
        image = Image.open(os.path.join(self.image_dir, name)).convert("RGB")
        mask = Image.open(os.path.join(self.mask_dir, name)).convert("L")

        # 2. 缩放到统一大小。原图用双线性插值，掩码必须用最近邻（否则会出现灰色边缘）
        size = (self.image_size, self.image_size)
        image = image.resize(size, Image.BILINEAR)
        mask = mask.resize(size, Image.NEAREST)

        # 3. 数据增强：随机左右翻转（训练时用，验证时不用）
        if self.augment and random.random() < 0.5:
            image = image.transpose(Image.FLIP_LEFT_RIGHT)
            mask = mask.transpose(Image.FLIP_LEFT_RIGHT)

        # 4. 转成 numpy，再转成 torch 张量
        image = np.array(image, dtype=np.float32) / 255.0    # 像素值缩到 0~1
        mask = np.array(mask, dtype=np.float32) / 255.0
        mask = (mask > 0.5).astype(np.float32)               # 变成 0 或 1

        image = image.transpose(2, 0, 1)      # HWC -> CHW，PyTorch 要通道在前
        mask = mask[None, :, :]               # 加一个通道维：H,W -> 1,H,W

        return torch.from_numpy(image), torch.from_numpy(mask)
