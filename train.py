import os
import random
import time
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import config
from unet import UNet
from dataset import SegDataset


def dice_loss(logits, masks, eps=1e-6):

    probs = torch.sigmoid(logits)                      # 把分数变成 0~1 的概率
    inter = (probs * masks).sum(dim=(1, 2, 3))         # 预测和真实重叠的面积
    union = probs.sum(dim=(1, 2, 3)) + masks.sum(dim=(1, 2, 3))
    dice = (2 * inter + eps) / (union + eps)
    return (1 - dice).mean()


def main():
    # 固定随机种子，保证每次跑结果差不多
    torch.manual_seed(0)
    random.seed(0)
    np.random.seed(0)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("使用设备:", device)

    # ---------------- 1. 数据 ----------------
    train_image_dir = os.path.join(config.DATA_DIR, "train", "images")
    train_mask_dir = os.path.join(config.DATA_DIR, "train", "masks")
    val_image_dir = os.path.join(config.DATA_DIR, "val", "images")
    val_mask_dir = os.path.join(config.DATA_DIR, "val", "masks")


    train_dataset = SegDataset(train_image_dir, train_mask_dir, config.IMAGE_SIZE, augment=True)
    val_dataset = SegDataset(val_image_dir, val_mask_dir, config.IMAGE_SIZE, augment=False)

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        num_workers=config.NUM_WORKERS,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=config.NUM_WORKERS,
    )
    print("训练图片 %d 张，验证图片 %d 张" % (len(train_dataset), len(val_dataset)))

    # ---------------- 2. 模型、损失、优化器 ----------------
    model = UNet(in_channels=3, num_classes=1, base=config.BASE_CHANNELS).to(device)
    bce = nn.BCEWithLogitsLoss()                       # 逐像素二分类损失
    optimizer = torch.optim.Adam(model.parameters(), lr=config.LR)

    os.makedirs(config.CHECKPOINT_DIR, exist_ok=True)
    best_dice = -1.0

    # ---------------- 3. 开始训练 ----------------
    for epoch in range(1, config.EPOCHS + 1):
        start_time = time.time()

        # ---- 训练 ----
        model.train()
        train_loss = 0.0
        for images, masks in train_loader:
            images = images.to(device)
            masks = masks.to(device)

            logits = model(images)                     # 前向：得到每个像素的分数
            loss = bce(logits, masks) + dice_loss(logits, masks)

            optimizer.zero_grad()                      # 清空上一步的梯度
            loss.backward()                            # 反向传播，算梯度
            optimizer.step()                           # 更新参数

            train_loss += loss.item() * images.size(0)
        train_loss = train_loss / len(train_dataset)

        # ---- 验证 ----
        model.eval()
        val_loss = 0.0
        inter_sum = 0.0
        pred_sum = 0.0
        mask_sum = 0.0
        with torch.no_grad():
            for images, masks in val_loader:
                images = images.to(device)
                masks = masks.to(device)

                logits = model(images)
                loss = bce(logits, masks) + dice_loss(logits, masks)
                val_loss += loss.item() * images.size(0)

                preds = (torch.sigmoid(logits) > 0.5).float()
                inter_sum += (preds * masks).sum().item()
                pred_sum += preds.sum().item()
                mask_sum += masks.sum().item()

        val_loss = val_loss / len(val_dataset)
        val_dice = (2 * inter_sum + 1e-6) / (pred_sum + mask_sum + 1e-6)

        print(
            "第 %2d/%d 轮 | 训练 loss %.4f | 验证 loss %.4f | 验证 Dice %.4f | %.1f 秒"
            % (epoch, config.EPOCHS, train_loss, val_loss, val_dice, time.time() - start_time)
        )

        # ---- 保存表现最好的模型 ----
        if val_dice > best_dice:
            best_dice = val_dice
            save_path = os.path.join(config.CHECKPOINT_DIR, "best.pt")
            torch.save(
                {
                    "model": model.state_dict(),
                    "epoch": epoch,
                    "val_dice": val_dice,
                    "base": config.BASE_CHANNELS,
                    "in_channels": 3,
                    "num_classes": 1,
                },
                save_path,
            )
            print("新的最好成绩，模型已保存到 " + save_path)

    # ---------------- 4. 收尾 ----------------
    print("训练结束，最好的一次验证 Dice = %.4f" % best_dice)
    print("模型文件：" + os.path.abspath(os.path.join(config.CHECKPOINT_DIR, "best.pt")))


if __name__ == "__main__":
    main()
