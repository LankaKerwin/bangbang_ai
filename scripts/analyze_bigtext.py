# -*- coding: utf-8 -*-
"""
中央大字公告 OCR 测试 (奖励时间/Boss公告) — 颜色无关, 读文本
==========================================================
【奖励时间】大字颜色会变化(本例粉色) → 不用颜色模板, 直接用 RapidOCR 读文本。
本脚本: 读中央大字带 → 多种预处理 → OCR → 打印文本, 验证能不能读出"奖励时间"。

用法:
    python scripts/analyze_bigtext.py            # 默认读 大字公告 最新一张
    python scripts/analyze_bigtext.py --img 某图.png
"""
import os
import sys
import glob
import argparse

import cv2
import numpy as np

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_LIBS = os.path.join(BASE, "libs_ocr")
if os.path.isdir(_LIBS) and _LIBS not in sys.path:
    sys.path.insert(0, _LIBS)

# 中央大字带 ROI (占屏比例): 大字横跨中央 y约25-65%
ROI = (0.05, 0.95, 0.20, 0.75)

_engine = None


def _get_engine():
    global _engine
    if _engine is None:
        from rapidocr_onnxruntime import RapidOCR
        _engine = RapidOCR()
    return _engine


def imread_u(p):
    with open(p, "rb") as f:
        return cv2.imdecode(np.fromfile(f, dtype=np.uint8), cv2.IMREAD_COLOR)


def _ocr_texts(bgr):
    out = _get_engine()(bgr)
    res = out[0] if isinstance(out, tuple) else out
    return [it[1] for it in (res or []) if len(it) >= 2]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--img", default=None)
    ap.add_argument("--rx0", type=float, default=ROI[0])
    ap.add_argument("--rx1", type=float, default=ROI[1])
    ap.add_argument("--ry0", type=float, default=ROI[2])
    ap.add_argument("--ry1", type=float, default=ROI[3])
    args = ap.parse_args()

    img = args.img
    if img is None:
        cands = sorted(glob.glob(os.path.join(r"D:\AI\临时文件\大字公告", "*.png")),
                       key=os.path.getmtime)
        img = cands[-1] if cands else None
    if not img or not os.path.exists(img):
        print("NO_IMG 大字公告文件夹没有图, 用 --img 指定")
        return

    bgr = imread_u(img)
    h, w = bgr.shape[:2]
    x0, x1 = int(w * args.rx0), int(w * args.rx1)
    y0, y1 = int(h * args.ry0), int(h * args.ry1)
    roi = bgr[y0:y1, x0:x1]
    print("IMG %dx%d | ROI(%d,%d)-(%d,%d) %dx%d  [%s]" % (w, h, x0, y0, x1, y1,
          roi.shape[1], roi.shape[0], os.path.basename(img)))

    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    # 大字超大的话要缩小到 rec 友好尺度(字高~30-50px), 多档尝试
    cands = {}
    for fx in (1.6, 0.5, 0.3, 0.18):
        b = cv2.resize(roi, None, fx=fx, fy=fx, interpolation=cv2.INTER_CUBIC)
        g = cv2.resize(gray, None, fx=fx, fy=fx, interpolation=cv2.INTER_CUBIC)
        cands["彩色x%.2f" % fx] = b
        cands["反色x%.2f" % fx] = cv2.cvtColor(255 - g, cv2.COLOR_GRAY2BGR)

    for name, b in cands.items():
        try:
            texts = _ocr_texts(b)
        except Exception as e:
            print(f"[{name}] OCR异常: {e}")
            continue
        print(f"[{name}] → {texts}")

    # 大字像素占比(触发用参考): 亮 + 高饱和彩色
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    light = (gray > 180).astype(np.uint8)
    over_s = (hsv[:, :, 1] > 120).astype(np.uint8)
    print("亮字占比=%.3f / 高饱和占比=%.3f" % (float(light.mean()), float(over_s.mean())))


if __name__ == "__main__":
    main()
