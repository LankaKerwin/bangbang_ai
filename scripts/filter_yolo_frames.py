"""
枪豆人 YOLO 截图预筛选
用现有模型把"疑似有敌人"的图挑出来，减少人工标注量。

用法：
    python scripts/filter_yolo_frames.py --input yolo_data/images/to_label --model <best.pt>

输出：
    input/has_target/     检测到目标的图（去这里标注）
    input/empty_sample/   空图抽样保留（负样本，降低误检）
    其余空图删除
"""
import os
import random
import shutil
import argparse
from ultralytics import YOLO

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_MODEL = os.path.join(BASE_DIR, "models", "yolo", "bb_yolov8s", "weights", "best.pt")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default=os.path.join(BASE_DIR, "yolo_data", "images", "to_label"))
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--conf", type=float, default=0.25, help="置信度阈值(调低避免漏检)")
    parser.add_argument("--keep_empty_ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)
    if not os.path.exists(args.model):
        print(f"找不到模型: {args.model}，先训练或指定 --model")
        return

    has_dir = os.path.join(args.input, "has_target")
    empty_dir = os.path.join(args.input, "empty_sample")
    os.makedirs(has_dir, exist_ok=True)
    os.makedirs(empty_dir, exist_ok=True)

    imgs = [f for f in os.listdir(args.input)
            if os.path.isfile(os.path.join(args.input, f))
            and f.lower().endswith((".jpg", ".jpeg", ".png", ".bmp"))]
    imgs.sort()
    if not imgs:
        print(f"{args.input} 里没有图片")
        return

    print(f"待筛选 {len(imgs)} 张 | 模型 {os.path.basename(args.model)} | conf={args.conf}")
    model = YOLO(args.model)

    has_list, empty_list = [], []
    batch = 16
    for i in range(0, len(imgs), batch):
        paths = [os.path.join(args.input, f) for f in imgs[i:i + batch]]
        results = model.predict(paths, conf=args.conf, verbose=False, imgsz=640)
        for fname, r in zip(imgs[i:i + batch], results):
            (has_list if len(r.boxes) > 0 else empty_list).append(fname)
        print(f"  已处理 {min(i + batch, len(imgs))}/{len(imgs)} "
              f"(有目标 {len(has_list)} / 无目标 {len(empty_list)})")

    for f in has_list:
        shutil.move(os.path.join(args.input, f), os.path.join(has_dir, f))
    random.shuffle(empty_list)
    keep = int(len(empty_list) * args.keep_empty_ratio)
    for f in empty_list[:keep]:
        shutil.move(os.path.join(args.input, f), os.path.join(empty_dir, f))
    for f in empty_list[keep:]:
        try:
            os.remove(os.path.join(args.input, f))
        except OSError:
            pass

    print("=" * 50)
    print(f"有目标(待标注) {len(has_list)} 张 → {has_dir}")
    print(f"保留空图(负样本) {keep} 张 → {empty_dir}")
    print("下一步：labelme 标注 has_target，标完跑 labelme_to_yolo.py")


if __name__ == "__main__":
    main()
