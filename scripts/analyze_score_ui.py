# -*- coding: utf-8 -*-
"""
右上角 UI 静态分析: 把"钻石"(彩色图标) 与 "分数"(白字数字) 分区找出来
==================================================================
不抓屏, 直接读已有的截图(PNG) → 裁右上角 → 按列带统计 亮字比例/蓝色比例,
把像素事实转成文本直方图, 用于确定两个显示各自的 ROI。

用法:
    python scripts/analyze_score_ui.py                # 自动取 临时文件\分数 最新一张
    python scripts/analyze_score_ui.py --img 某图.png
    python scripts/analyze_score_ui.py --rx0 55 --rx1 100 --cols 24
"""
import os
import sys
import glob
import argparse

import cv2
import numpy as np


def imread_u(path):
    """中文路径安全读图: open+imdecode(照项目惯例)。"""
    with open(path, "rb") as f:
        return cv2.imdecode(np.frombuffer(f.read(), np.uint8), cv2.IMREAD_COLOR)


def save_img(path, img):
    ok, buf = cv2.imencode(".png", img)
    if not ok:
        return False
    with open(path, "wb") as f:
        f.write(buf.tobytes())
    return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--img", default=None, help="图片路径(默认取 临时文件\\分数 最新一张)")
    ap.add_argument("--rx0", type=float, default=55.0, help="右上框左边界%")
    ap.add_argument("--rx1", type=float, default=100.0)
    ap.add_argument("--ry0", type=float, default=0.0)
    ap.add_argument("--ry1", type=float, default=12.0)
    ap.add_argument("--cols", type=int, default=24)
    ap.add_argument("--outdir", default=None, help="标注图输出目录(默认同图片目录\\_ui_anno)")
    args = ap.parse_args()

    img = args.img
    if img is None:
        cands = sorted(glob.glob(os.path.join(r"D:\AI\临时文件\分数", "*.png")),
                       key=os.path.getmtime)
        img = cands[-1] if cands else None
    if not img or not os.path.exists(img):
        print("NO_IMG 没找到图片; 用 --img 指定")
        return

    bgr = imread_u(img)
    if bgr is None:
        print("READ_FAIL 无法解码:", img)
        return
    h, w = bgr.shape[:2]
    x0, x1 = int(w * args.rx0 / 100), int(w * args.rx1 / 100)
    y0, y1 = int(h * args.ry0 / 100), int(h * args.ry1 / 100)
    roi = bgr[y0:y1, x0:x1]
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    g = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    rh, rw = g.shape[:2]

    print("IMG %dx%d | ROI(%d,%d)-(%d,%d) = %dx%d  [%s]" % (w, h, x0, y0, x1, y1, rw, rh, img))
    print("col | xpx    | bright | blue  | 亮柱")
    bw = rw / args.cols
    for c in range(args.cols):
        xa, xb = int(c * bw), min(int((c + 1) * bw), rw)
        gcol = g[:, xa:xb]
        hcol = hsv[:, xa:xb, 0]
        scol = hsv[:, xa:xb, 1]
        vcol = hsv[:, xa:xb, 2]
        bright = float((gcol > 200).mean())
        blue = float(((hcol >= 90) & (hcol <= 130) & (scol > 70) & (vcol > 70)).mean())
        bar = "#" * int(bright * 60)
        print("%3d | %5d | %6.3f | %5.3f | %s" % (c, xa, bright, blue, bar))

    # 水平投影: 整行亮像素 与 蓝色像素 的行范围, 判断上下分层
    bm = (g > 200).astype(np.uint8)
    um = ((hsv[:, :, 0] >= 90) & (hsv[:, :, 0] <= 130) & (hsv[:, :, 1] > 70) & (hsv[:, :, 2] > 70)).astype(np.uint8)
    brow = bm.sum(axis=1)
    urow = um.sum(axis=1)
    brng = np.flatnonzero(brow)
    urng = np.flatnonzero(urow)
    print("bright 行范围:", (int(brng[0]), int(brng[-1])) if brng.size else "无")
    print("blue   行范围:", (int(urng[0]), int(urng[-1])) if urng.size else "无")

    outdir = args.outdir or os.path.join(os.path.dirname(img) or ".", "_ui_anno")
    os.makedirs(outdir, exist_ok=True)
    annot = roi.copy()
    for c in range(args.cols):
        xa = int(c * bw)
        cv2.line(annot, (xa, 0), (xa, rh), (0, 255, 0), 1)
    op = os.path.join(outdir, "cols_" + os.path.basename(img))
    save_img(op, annot)
    print("标注图(列带):", op)


if __name__ == "__main__":
    main()
