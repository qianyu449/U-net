# ---------------- 数据 ----------------
DATA_DIR = "data"        # 数据文件夹
IMAGE_SIZE = 128         # 统一把图片缩放到 128x128（必须是 16 的倍数，原因见 README）
DEMO_COUNT = 64          # 演示数据：训练集图片数量
DEMO_VAL_COUNT = 16      # 演示数据：验证集图片数量

# ---------------- 训练 ----------------
BATCH_SIZE = 8
EPOCHS = 10
LR = 0.001               # 学习率
BASE_CHANNELS = 32       # 第一层的通道数，后面每层翻倍。电脑慢就改小，比如 16
NUM_WORKERS = 0          # Windows 下建议保持 0

# ---------------- 保存 ----------------
CHECKPOINT_DIR = "model"  # 训练好的模型存这里
OUTPUT_DIR = "outputs"           # 预测结果存这里
