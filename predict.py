"""用训练好的模型做预测，并保存结果图片。

运行方式：
    python predict.py                    # 预测 data/test/images 下的所有图片
    python predict.py 图片路径            # 预测你自己的单张图

结果会保存到 outputs/ 文件夹：
    xxx_mask.png     模型预测的黑白掩码（白 = 目标）
    xxx_compare.png  对比图：原图 | 红色叠加 | 预测掩码 (如果有真实掩码，也会拼上去)
"""

# Windows 上偶尔会出现 "OMP: Error #15" 的报错，加这一行可以避开
import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import sys
import glob

import numpy as np
from PIL import Image, ImageDraw
import torch

import config
from unet import UNet

def load_image(path, size):
    """读一张图，缩放、转成 0~1 的张量，形状变成 1x3xHxW。"""
    image = Image.open(path).convert("RGB").resize((size, size), Image.BILINEAR)
    array = np.array(image, dtype=np.float32) / 255.0
    tensor = torch.from_numpy(array.transpose(2, 0, 1))[None, :, :, :]
    return image, tensor

def make_overlay(image, mask):
    """把预测区域涂成半透明红色，盖在原图上。"""
    array = np.array(image).astype(np.float32)
    array[mask] = array[mask] * 0.5 + np.array([255, 0, 0], dtype=np.float32) * 0.5
    return Image.fromarray(array.clip(0, 255).astype(np.uint8))

def add_title(image, text):
    """在图片左上角写一行小字，方便区分每一格是什么。"""
    canvas = image.copy()
    draw = ImageDraw.Draw(canvas)
    draw.rectangle([0, 0, len(text) * 7 + 8, 16], fill=(0, 0, 0))
    draw.text((4, 2), text, fill=(255, 255, 0))
    return canvas

def process_single_image(image_path, mask_path, model, device):
    """处理单张图片并保存结果"""
    # 预测
    image, tensor = load_image(image_path, config.IMAGE_SIZE)
    tensor = tensor.to(device)

    with torch.no_grad():
        logits = model(tensor)
        prob = torch.sigmoid(logits)                 # 每个像素是目标的概率
        pred = (prob > 0.5)[0, 0].cpu().numpy()      # 二值掩码，True/False

    # 保存结果
    os.makedirs(config.OUTPUT_DIR, exist_ok=True)
    stem = os.path.splitext(os.path.basename(image_path))[0]

    # 黑白掩码图
    mask_image = Image.fromarray((pred * 255).astype(np.uint8))
    mask_path_out = os.path.join(config.OUTPUT_DIR, stem + "_mask.png")
    mask_image.save(mask_path_out)

    # 拼一张对比图
    panels = [add_title(image, "image"), add_title(make_overlay(image, pred), "prediction")]

    if mask_path and os.path.isfile(mask_path):
        gt = Image.open(mask_path).convert("L").resize((config.IMAGE_SIZE, config.IMAGE_SIZE), Image.NEAREST)
        gt_array = (np.array(gt) > 127).astype(np.float32)
        gt_image = Image.fromarray((gt_array * 255).astype(np.uint8)).convert("RGB")

        # 顺便算一下 Dice
        inter = (pred & (gt_array > 0.5)).sum()
        dice = (2 * inter + 1e-6) / (pred.sum() + gt_array.sum() + 1e-6)
        print(f"  [{stem}] Dice = {dice:.4f}")
        panels.append(add_title(gt_image, "ground truth"))

    compare = Image.new("RGB", (config.IMAGE_SIZE * len(panels), config.IMAGE_SIZE))
    for i, panel in enumerate(panels):
        compare.paste(panel, (i * config.IMAGE_SIZE, 0))
    compare_path = os.path.join(config.OUTPUT_DIR, stem + "_compare.png")
    compare.save(compare_path)

    return mask_path_out, compare_path

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ---------------- 1. 加载模型 ----------------
    checkpoint_path = os.path.join(config.CHECKPOINT_DIR, "best.pt")
    if not os.path.isfile(checkpoint_path):
        print("没有找到模型文件，请先运行 python train.py")
        return

    checkpoint = torch.load(checkpoint_path, map_location=device)
    model = UNet(
        in_channels=checkpoint.get("in_channels", 3),
        num_classes=checkpoint.get("num_classes", 1),
        base=checkpoint.get("base", config.BASE_CHANNELS),
    )
    model.load_state_dict(checkpoint["model"])
    model.to(device)
    model.eval()
    print("模型来自第 %d 轮，验证 Dice = %.4f" % (checkpoint["epoch"], checkpoint["val_dice"]))
    print("-" * 50)

    # ---------------- 2. 确定要预测的图片列表 ----------------
    tasks = []

    if len(sys.argv) > 1:
        # 如果命令行传了参数，就预测指定的单张图
        image_path = sys.argv[1]
        if not os.path.isfile(image_path):
            print(f"找不到文件：{image_path}")
            return
        tasks.append((image_path, None))
    else:
        # 否则，预测 test/images 下的所有图片
        test_image_dir = os.path.join(config.DATA_DIR, "test", "images")
        if not os.path.isdir(test_image_dir):
            print(f"没有找到测试集文件夹：{test_image_dir}")
            return

        valid_exts = ('.png', '.jpg', '.jpeg', '.bmp', '.tif')
        image_names = [f for f in os.listdir(test_image_dir) if f.lower().endswith(valid_exts)]

        if not image_names:
            print(f"测试集文件夹中没有找到图片：{test_image_dir}")
            return

        print(f"在测试集中找到 {len(image_names)} 张图片，开始预测...")
        for name in image_names:
            img_p = os.path.join(test_image_dir, name)
            # 测试集可能没有 masks 文件夹，或者有；有就顺带算一下 Dice
            mask_p = os.path.join(config.DATA_DIR, "test", "masks", name)
            tasks.append((img_p, mask_p if os.path.isfile(mask_p) else None))

    # ---------------- 3. 循环执行预测 ----------------
    for img_path, mask_path in tasks:
        mask_out, compare_out = process_single_image(img_path, mask_path, model, device)

    print("-" * 50)
    print(f"全部预测完成！共处理 {len(tasks)} 张图片。")
    print(f"结果已全部保存至：{os.path.abspath(config.OUTPUT_DIR)}")

if __name__ == "__main__":
    main()