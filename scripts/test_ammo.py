# -*- coding: utf-8 -*-
"""
弹药检测对拍测试 — 用已知真值截图验证 ammo_detect 是否工作
用法:
    python scripts/test_ammo.py "D:\临时文件\弹药标定\3发.png" "D:\临时文件\弹药标定\0发_装填中.png"
    python scripts/test_ammo.py --image xxx.png --expected 2 --player "640,800"
    python scripts/test_ammo.py --dir D:\临时文件\弹药标定 --save 输出目录

期望值默认从文件名取数字("3发.png"→3, "0发_装填中.png"→0)。
自动用 YOLO 找玩家船(若给了 --model)；找不到就用 --player 或画面中心。
每张图会打印 检测弹药数/期望/半径/角度，并可选存标注图供人看。
"""
import os
import re
import sys
import glob
import argparse

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ammo_detect import detect_ammo  # noqa: E402

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


def expected_from_name(name):
    m = re.search(r"(\d+)\s*发", name)
    if m:
        return int(m.group(1))
    m = re.search(r"_(\d+)(?:\.|$)", name)
    if m:
        return int(m.group(1))
    return None


def find_player(img, model):
    h, w = img.shape[:2]
    if model is not None:
        r = model(img, imgsz=640, conf=0.25, verbose=False)
        bx = r[0].boxes
        if bx is not None and len(bx) > 0:
            for b in bx:
                if int(b.cls[0]) == 0:
                    x1, y1, x2, y2 = [float(v) for v in b.xyxy[0].tolist()]
                    return ((x1 + x2) / 2, (y1 + y2) / 2)
    return (w / 2, h / 2)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("images", nargs="*", help="图片路径")
    parser.add_argument("--image", default=None)
    parser.add_argument("--dir", default=None)
    parser.add_argument("--expected", type=int, default=None)
    parser.add_argument("--player", default=None, help="'x,y'，不给则自动YOLO找船")
    parser.add_argument("--model", default=MODEL_PATH)
    parser.add_argument("--save", default=None, help="标注图输出目录")
    args = parser.parse_args()

    model = None
    if os.path.exists(args.model):
        from ultralytics import YOLO
        model = YOLO(args.model)

    files = list(args.images)
    if args.image:
        files.append(args.image)
    if args.dir:
        files += sorted(glob.glob(os.path.join(args.dir, "*")))
    files = [f for f in files if f.lower().endswith((".png", ".jpg", ".jpeg"))]
    if not files:
        print("没有图片。用法示例见文件头。")
        return

    single_expected = args.expected if len(files) == 1 else None
    ok_cnt = total = 0
    for f in files:
        img = imread_u(f)
        if img is None:
            print(f"[跳过] 读不了: {f}")
            continue
        if args.player:
            x, y = args.player.split(",")
            pbox = (float(x), float(y))
        else:
            pbox = find_player(img, model)

        info = detect_ammo(img, pbox,
                           debug_draw=(img.copy() if args.save else None))
        exp = single_expected if single_expected is not None else expected_from_name(os.path.basename(f))
        tag = ""
        if exp is not None:
            total += 1
            match = info["ammo"] == exp
            ok_cnt += match
            tag = "✅" if match else "❌"
        print(f"{tag} {os.path.basename(f)}: 检测弹药={info['ammo']} (期望={exp}) "
              f"半径={info['radius']} 角度={info['angle_deg']}")
        if args.save and info["dots"]:
            os.makedirs(args.save, exist_ok=True)
            out_p = os.path.join(args.save, "anno_" + os.path.basename(f))
            imwrite_u(out_p, img)

    if total:
        print(f"\n对拍结果: {ok_cnt}/{total} 通过")


if __name__ == "__main__":
    main()
