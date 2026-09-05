"""
枪豆人 YOLO 自动预标注
用当前最强模型给未标注图片生成 labelme json（先框好，人只检查修正）。

用法：
    python scripts/auto_label_yolo.py --dir yolo_data/images/to_label/has_target --model <best.pt> --conf 0.3
"""
import os
import json
import argparse
from pathlib import Path
import cv2
from ultralytics import YOLO

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_MODEL = os.path.join(BASE_DIR, "models", "yolo", "bb_yolov8s", "weights", "best.pt")


def yolo_to_labelme(img_path, results, conf_threshold, names):
    img = cv2.imread(str(img_path))
    if img is None:
        return None
    ih, iw = img.shape[:2]
    shapes = []
    boxes = results[0].boxes
    if boxes is not None and len(boxes) > 0:
        for box in boxes:
            if float(box.conf[0]) < conf_threshold:
                continue
            cid = int(box.cls[0])
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            shapes.append({
                "label": names.get(cid, f"class_{cid}"),
                "points": [[round(x1), round(y1)], [round(x2), round(y2)]],
                "group_id": None, "description": "",
                "shape_type": "rectangle", "flags": {},
            })
    return {
        "version": "5.2.0.post4", "flags": {}, "shapes": shapes,
        "imagePath": os.path.basename(str(img_path)), "imageData": None,
        "imageHeight": ih, "imageWidth": iw,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", default=os.path.join(BASE_DIR, "yolo_data", "images", "to_label", "has_target"))
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--conf", type=float, default=0.3)
    args = parser.parse_args()

    img_dir = Path(args.dir)
    if not img_dir.exists():
        print(f"目录不存在: {img_dir}")
        return
    if not os.path.exists(args.model):
        print(f"找不到模型: {args.model}")
        return

    model = YOLO(args.model)
    names = {i: n for i, n in model.names.items()}  # id->token

    imgs = sorted([f for f in img_dir.iterdir()
                   if f.suffix.lower() in (".jpg", ".jpeg", ".png", ".bmp")
                   and not f.with_suffix(".json").exists()])
    print(f"待自动标注 {len(imgs)} 张 (conf={args.conf}) ...")

    done = empty = 0
    for p in imgs:
        try:
            r = model(str(p), imgsz=640, conf=args.conf, iou=0.5, verbose=False)
            data = yolo_to_labelme(p, r, args.conf, names)
            if data is None:
                continue
            with open(p.with_suffix(".json"), "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
            done += 1
            empty += (len(data["shapes"]) == 0)
        except Exception as e:
            print(f"  [错误] {p.name}: {e}")

    print(f"\n完成 {done} 张，其中空标注 {empty} 张")
    print("下一步：labelme 打开该目录检查修正(漏的补、错的改、多余的删)")


if __name__ == "__main__":
    main()
