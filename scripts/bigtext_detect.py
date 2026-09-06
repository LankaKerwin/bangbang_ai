# -*- coding: utf-8 -*-
"""
bigtext_detect.py — 中央大字公告检测 (Boss击杀事件 = 奖励时间)
=============================================================
Boss 击败后屏幕中央出现大字公告「奖励时间」(颜色可变, 样例为粉色)。
颜色无关: 直接彩色图缩小到 rec 友好尺度 → RapidOCR 读文本 → 匹配。

复用 score_ocr._get_engine 单例(避免重复加载模型)。

用法:
    from bigtext_detect import detect_reward_time
    hit = detect_reward_time(bgr_frame)      # → bool
"""
import os
import re

import cv2

from score_ocr import _get_engine

# 中央大字带 ROI (占屏比例): 大字横跨中央, y约20-75% (标定 2026-09-06 奖励时间样例)
ROI = (0.05, 0.95, 0.20, 0.75)
SCALE = 0.35            # 大字(高~880px) 缩小到 rec 友好(~300px); 0.5/0.3/0.18 均验证可读
TARGET_TEXT = "奖励时间"


def _clean(s):
    return re.sub(r"[\s!！。·、,，\-—~～:：]", "", s)


def detect_reward_time(bgr):
    """检测本帧中央大字是否为「奖励时间」。返回 bool。"""
    h, w = bgr.shape[:2]
    x0, x1, y0, y1 = ROI
    roi = bgr[int(h * y0):int(h * y1), int(w * x0):int(w * x1)]
    if roi.size == 0:
        return False
    big = cv2.resize(roi, None, fx=SCALE, fy=SCALE, interpolation=cv2.INTER_CUBIC)
    try:
        out = _get_engine()(big)
    except Exception:
        return False
    res = out[0] if isinstance(out, tuple) else out
    joined = "".join(it[1] for it in (res or []) if len(it) >= 2)
    return _clean(joined) == TARGET_TEXT or TARGET_TEXT in _clean(joined)


if __name__ == "__main__":
    import glob
    import numpy as np

    def imread_u(p):
        with open(p, "rb") as f:
            return cv2.imdecode(np.fromfile(p, dtype=np.uint8), cv2.IMREAD_COLOR)

    files = sorted(glob.glob(os.path.join(r"D:\AI\临时文件\大字公告", "*.png")),
                   key=os.path.getmtime)
    if not files:
        print("无图")
        raise SystemExit(0)
    for f in files:
        bgr = imread_u(f)
        print(f"{os.path.basename(f)} → reward_time={detect_reward_time(bgr)}")
