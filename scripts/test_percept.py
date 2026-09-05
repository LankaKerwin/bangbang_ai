# -*- coding: utf-8 -*-
"""
test_percept.py — 感知(弹幕栅格/掉落物)对拍标注工具
用法:
    python scripts/test_percept.py --dir D:\临时文件\感知标定            # 批量
    python scripts/test_percept.py --dir ... --save 输出目录             # 存标注图
每张图打印: 敌方弹数 / 最危险扇区角度 / 掉落物列表；标注图圈出子弹(红)与掉落(蓝/橙)。
"""
import os
import glob
import argparse

import cv2
import numpy as np

import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from bullet_grid import danger_grid, grid_sector_angle, RINGS_PX  # noqa: E402
from drop_detect import detect_drops                              # noqa: E402

MODEL_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                          "models", "release", "bb_yolov8s_v1.pt")
if not os.path.exists(MODEL_PATH):
    MODEL_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                              "models", "yolo", "bb_yolov8s", "weights", "best.pt")


def imread_u(path):
    with open(path, "rb") as f:
        return cv2.imdecode(np.frombuffer(f.read(), np.uint8), cv2.IMREAD_COLOR)


def imwrite_u(path, img):
    ok, buf = cv2.imencode(".png", img)
    if ok:
        with open(path, "wb") as f:
            f.write(buf.tobytes())


def find_player(img, model):
    h, w = img.shape[:2]
    r = model(img, imgsz=640, conf=0.25, verbose=False)
    bx = r[0].boxes
    if bx is not None and len(bx) > 0:
        for b in bx:
            if int(b.cls[0]) == 0:
                x1, y1, x2, y2 = [float(v) for v in b.xyxy[0].tolist()]
                return ((x1 + x2) / 2, (y1 + y2) / 2)
    return (w / 2, h / 2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True, help="放截图/录屏抽帧的目录")
    ap.add_argument("--save", default=None, help="标注图输出目录")
    args = ap.parse_args()

    from ultralytics import YOLO
    model = YOLO(MODEL_PATH) if os.path.exists(MODEL_PATH) else None

    files = sorted(f for f in glob.glob(os.path.join(args.dir, "*"))
                   if f.lower().endswith((".png", ".jpg", ".jpeg")))
    print(f"共 {len(files)} 张")
    for f in files:
        img = imread_u(f)
        if img is None:
            continue
        h, w = img.shape[:2]
        enemy_boxes = []
        pbox = (w / 2, h / 2)
        if model is not None:
            r = model(img, imgsz=640, conf=0.25, verbose=False)
            bx = r[0].boxes
            if bx is not None and len(bx) > 0:
                for b in bx:
                    x1, y1, x2, y2 = [float(v) for v in b.xyxy[0].tolist()]
                    if int(b.cls[0]) == 0:
                        pbox = ((x1 + x2) / 2, (y1 + y2) / 2)
                    else:
                        enemy_boxes.append((x1, y1, x2, y2))
        vis = img.copy() if args.save else None
        bullets, grid = danger_grid(img, pbox, debug_draw=vis)
        drops = detect_drops(img, enemy_boxes=enemy_boxes, debug_draw=vis)
        worst = int(np.argmax(grid)) if grid.max() > 0 else None
        worst_deg = grid_sector_angle(worst // len(RINGS_PX)) if worst is not None else None
        dtxt = ",".join(d["type"] for d in drops[:8]) or "-"
        print(f"{os.path.basename(f)}: 敌弹={len(bullets)} "
              f"最危扇区角={worst_deg if worst_deg is not None else '-'} 掉落=[{dtxt}]")
        if args.save:
            os.makedirs(args.save, exist_ok=True)
            cv2.putText(vis, f"bullets={len(bullets)} drops={len(drops)}", (20, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 255), 3)
            imwrite_u(os.path.join(args.save, "p_" + os.path.basename(f)), vis)


if __name__ == "__main__":
    main()
