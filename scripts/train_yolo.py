"""
枪豆人 YOLO 训练脚本 (YOLOv8)
用法：
    python scripts/train_yolo.py

说明：
- 用官方预训练权重迁移学习。**只使用本地权重文件，绝不联网自动下载**
  （国内网络访问 GitHub 会 SSL 失败）。找不到权重会打印指引后退出。
- 数据来自 yolo_data/data.yaml（45 类 v1）
"""
import os
from ultralytics import YOLO

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_YAML = os.path.join(BASE_DIR, "yolo_data", "data.yaml")
PROJECT_DIR = os.path.join(BASE_DIR, "models", "yolo")

# 本地预训练权重候选（按优先级；都不存在就提示，不走自动下载）
TL_CANDIDATES = [
    os.path.join(BASE_DIR, "yolov8s.pt"),                 # 1) 项目根
    r"D:\AI\AI train\tlou2_ai\yolov8s.pt",                # 2) tlou2 已下载的
    r"D:\AI\AI train\tlou2_ai\yolov8n.pt",                # 3) tlou2 nano 备选
]
TL_W = next((p for p in TL_CANDIDATES if os.path.exists(p)), None)


def main():
    if TL_W is None:
        print("=" * 55)
        print("❌ 找不到本地预训练权重 yolov8s.pt！")
        print("   请任选其一：")
        print(f"   A. 复制 tlou2 的:  Copy-Item 'D:\\AI\\AI train\\tlou2_ai\\yolov8s.pt' "
              f"'{os.path.join(BASE_DIR, 'yolov8s.pt')}'")
        print("   B. 手动下载后放到项目根目录（如浏览器/镜像下载 yolov8s.pt）")
        print("   脚本不会联网自动下载（避免 SSL/网络问题）。")
        print("=" * 55)
        return

    print(f"使用预训练权重: {TL_W}")
    model = YOLO(TL_W)

    results = model.train(
        data=DATA_YAML,
        epochs=150,          # 首轮可以多训；形态固定的类收敛快
        imgsz=640,           # 手/花目标偏小，若小目标漏检可试 960（吃显存）
        batch=16,            # 8G 显存 nano 可 32，small 建议 16
        device=0,
        workers=4,
        patience=25,
        project=PROJECT_DIR,
        name="bb_yolov8s",
        exist_ok=True,
        # 数据增强
        hsv_h=0.02,          # 敌我靠形态为主，色相扰动小一点
        hsv_s=0.4,           # 饱和度扰大点：抗黄昏/夜晚色调变化
        hsv_v=0.4,           # 明度扰大点：抗黑夜
        flipud=0.5,          # 该游戏画面上下翻转也合理(敌可从各方向来)
        fliplr=0.5,
        mosaic=1.0,
        mixup=0.1,
    )

    print("\n=== 验证最佳模型 ===")
    metrics = model.val()
    print(f"mAP50: {metrics.box.map50:.4f}")
    print(f"mAP50-95: {metrics.box.map:.4f}")
    print(f"\n模型保存在: {os.path.join(PROJECT_DIR, 'bb_yolov8s', 'weights', 'best.pt')}")
    print("下一步：python scripts/test_yolo.py 开小窗实时预览检测效果")


if __name__ == "__main__":
    main()
