import torch
import torch.nn as nn


def double_conv(in_ch, out_ch):
    # 这是一个卷积层
    # 连着做两次 [3x3 卷积 + BN + ReLU]。
    # 卷积用 padding=1，这样图片长宽不会变小。
    # BN（批归一化）让训练更稳、收敛更快。
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(inplace=True),

        nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, bias=False),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(inplace=True),
    )

class UNet(nn.Module):

    def __init__(self, in_channels=3, num_classes=1, base=32):
        super().__init__()

        # ===== 左边：编码器（每层先提特征，再用池化把长宽减半）=====
        self.enc1 = double_conv(in_channels, base)          # 1/1 大小
        self.enc2 = double_conv(base, base * 2)             # 1/2 大小
        self.enc3 = double_conv(base * 2, base * 4)         # 1/4 大小
        self.enc4 = double_conv(base * 4, base * 8)         # 1/8 大小
        self.pool = nn.MaxPool2d(2)                         # 2x2 最大池化，长宽减半

        # ===== U 的底部 =====
        self.center = double_conv(base * 8, base * 16)      # 1/16 大小

        # ===== 右边：解码器（先放大一倍，再拼接，再提特征）=====
        self.up4 = nn.ConvTranspose2d(base * 16, base * 8, kernel_size=2, stride=2)
        self.dec4 = double_conv(base * 16, base * 8)

        self.up3 = nn.ConvTranspose2d(base * 8, base * 4, kernel_size=2, stride=2)
        self.dec3 = double_conv(base * 8, base * 4)

        self.up2 = nn.ConvTranspose2d(base * 4, base * 2, kernel_size=2, stride=2)
        self.dec2 = double_conv(base * 4, base * 2)

        self.up1 = nn.ConvTranspose2d(base * 2, base, kernel_size=2, stride=2)
        self.dec1 = double_conv(base * 2, base)

        # 最后用 1x1 卷积，把通道数变成 num_classes（每个像素得到一个分数）
        self.out = nn.Conv2d(base, num_classes, kernel_size=1)

        # 初始化权重，让训练一开始就比较好
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        if x.shape[-1] % 16 != 0 or x.shape[-2] % 16 != 0:
            raise ValueError("输入图片的长宽必须是 16 的倍数，比如 128、256")

        # ---------- 左边：一路往下，把每层结果都存下来 ----------
        e1 = self.enc1(x)                       # 128 x 128, 32 通道
        e2 = self.enc2(self.pool(e1))           # 64  x 64,  64 通道
        e3 = self.enc3(self.pool(e2))           # 32  x 32,  128 通道
        e4 = self.enc4(self.pool(e3))           # 16  x 16,  256 通道
        center = self.center(self.pool(e4))     # 8   x 8,   512 通道

        # ---------- 右边：一路往上，每一步都和左边拼一下 ----------
        d4 = self.up4(center)                   # 放大到 16 x 16
        d4 = torch.cat([d4, e4], dim=1)         # 和左边 e4 拼起来，通道数变成 512
        d4 = self.dec4(d4)

        d3 = self.up3(d4)                       # 放大到 32 x 32
        d3 = torch.cat([d3, e3], dim=1)
        d3 = self.dec3(d3)

        d2 = self.up2(d3)                       # 放大到 64 x 64
        d2 = torch.cat([d2, e2], dim=1)
        d2 = self.dec2(d2)

        d1 = self.up1(d2)                       # 放大回 128 x 128
        d1 = torch.cat([d1, e1], dim=1)
        d1 = self.dec1(d1)

        # 输出的是"分数"（logits），还没经过 sigmoid，损失函数里会处理
        return self.out(d1)



    # if __name__ == "__main__":
    # model = UNet(in_channels=3, num_classes=1, base=32)
    # x = torch.randn(2, 3, 128, 128)   # 假装有 2 张 128x128 的彩色图
    # y = model(x)
    # print("输入形状:", tuple(x.shape))
    # print("输出形状:", tuple(y.shape))
    # print("参数量:", sum(p.numel() for p in model.parameters()) / 1e6, "M")
