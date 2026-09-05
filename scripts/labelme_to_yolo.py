"""
枪豆人 labelme JSON -> YOLO TXT 转换 + 训练/验证集划分

用法：
    python scripts/labelme_to_yolo.py                               # 只用默认 has_target
    python scripts/labelme_to_yolo.py --src 目录1,目录2,...          # 合并多批(逗号分隔)

说明：
- 默认标注目录: yolo_data/images/to_label/has_target（图片与 labelme json 同目录）
- --src 可指定一个或多个已标注目录；多次运行会【累积】并入 train/val，旧文件不删
- 自动 90% 训练 / 10% 验证
- 标签映射见 scripts/bb_classes.py（兼容中文/英文/大小写/别名）
- YOLO txt 输出: yolo_data/labels/train|val（yaml 里配的路径）
"""
import os
import sys
import json
import random
import shutil
import argparse

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from bb_classes import label_to_id, CN_NAMES  # noqa: E402

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
YOLO_DATA = os.path.join(BASE_DIR, "yolo_data")

SRC_IMG_DIR = os.path.join(YOLO_DATA, "images", "to_label", "has_target")  # json 同目录
TRAIN_IMG_DIR = os.path.join(YOLO_DATA, "images", "train")
VAL_IMG_DIR = os.path.join(YOLO_DATA, "images", "val")
TRAIN_LABEL_DIR = os.path.join(YOLO_DATA, "labels", "train")
VAL_LABEL_DIR = os.path.join(YOLO_DATA, "labels", "val")

VAL_RATIO = 0.1
RANDOM_SEED = 42
IMG_EXTS = (".jpg", ".jpeg", ".png", ".bmp")


def json_to_yolo_txt(json_path, img_w, img_h):
    """labelme json -> YOLO 行。矩形框转 cx,cy,w,h 归一化。"""
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    lines = []
    for shape in data.get("shapes", []):
        if shape.get("shape_type") != "rectangle":
            continue
        label = shape.get("label", "").strip()
        try:
            cid = label_to_id(label)
        except KeyError:
            print(f"  [警告] 未知标签 '{label}'，跳过（检查 bb_classes.py）")
            continue

        pts = shape["points"]
        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]
        x1, x2, y1, y2 = min(xs), max(xs), min(ys), max(ys)

        cx = (x1 + x2) / 2.0 / img_w
        cy = (y1 + y2) / 2.0 / img_h
        w = (x2 - x1) / img_w
        h = (y2 - y1) / img_h
        cx, cy = max(0.0, min(1.0, cx)), max(0.0, min(1.0, cy))
        w, h = max(0.0, min(1.0, w)), max(0.0, min(1.0, h))
        if w < 0.001 or h < 0.001:
            continue
        lines.append(f"{cid} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")
    return lines


def collect_images(src_dirs):
    """从多个目录收集 (目录, 文件名) 列表。"""
    items = []
    for d in src_dirs:
        if not os.path.isdir(d):
            print(f"[跳过] 目录不存在: {d}")
            continue
        for f in sorted(os.listdir(d)):
            if f.lower().endswith(IMG_EXTS):
                items.append((d, f))
    return items


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--src", default=None,
                        help="标注目录(逗号分隔多目录)；默认 yolo_data/images/to_label/has_target")
    args = parser.parse_args()

    if args.src:
        src_dirs = [p.strip() for p in args.src.split(",") if p.strip()]
    else:
        src_dirs = [SRC_IMG_DIR]

    random.seed(RANDOM_SEED)
    for d in (TRAIN_IMG_DIR, VAL_IMG_DIR, TRAIN_LABEL_DIR, VAL_LABEL_DIR):
        os.makedirs(d, exist_ok=True)

    items = collect_images(src_dirs)
    random.shuffle(items)
    if not items:
        print(f"{src_dirs} 里没有图片")
        return

    val_n = max(1, int(len(items) * VAL_RATIO))
    print(f"共 {len(items)} 张 (来源 {len(src_dirs)} 目录) → 训练 {len(items) - val_n} / 验证 {val_n}")
    stats = {"train": {}, "val": {}}

    for idx, (src_dir, name) in enumerate(items):
        split = "val" if idx < val_n else "train"
        dst_img = (VAL_IMG_DIR if split == "val" else TRAIN_IMG_DIR)
        dst_lab = (VAL_LABEL_DIR if split == "val" else TRAIN_LABEL_DIR)

        shutil.copy2(os.path.join(src_dir, name), os.path.join(dst_img, name))

        stem = os.path.splitext(name)[0]
        jpath = os.path.join(src_dir, stem + ".json")
        if os.path.exists(jpath):
            with open(jpath, "r", encoding="utf-8") as f:
                meta = json.load(f)
            iw = meta.get("imageWidth", 1280)
            ih = meta.get("imageHeight", 720)
            lines = json_to_yolo_txt(jpath, iw, ih)
        else:
            lines = []

        if lines:
            with open(os.path.join(dst_lab, stem + ".txt"), "w") as f:
                f.write("\n".join(lines) + "\n")
            for ln in lines:
                cid = int(ln.split()[0])
                stats[split][cid] = stats[split].get(cid, 0) + 1

    print("\n" + "=" * 50)
    for split in ("train", "val"):
        s = stats[split]
        print(f"【{split.upper()}】 各类框数：")
        for cid in sorted(s):
            print(f"   {cid:2d} {CN_NAMES.get(cid, '?'):<20} {s[cid]}")
    print("=" * 50)
    print("\n下一步：python scripts/train_yolo.py 开始训练")
    print("(空图目录 labels 里不会生成 txt，YOLO 会自动当负样本)")


if __name__ == "__main__":
    main()
