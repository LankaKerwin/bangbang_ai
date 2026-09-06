# -*- coding: utf-8 -*-
"""
score_ocr.py — 右上角分数实时 OCR (击杀奖励信号源)
=================================================
从游戏截图读右上角的【白色分数数字】 → int。分数在右上角 x≈90-98.5%,
y≈5-9.5%; 上方青绿小字 COMBO 已被 ROI 避开(不同 y 带)。分数=白色, 带千分位逗号。

关键: RapidOCR 默认检测对"稀疏小白字"会把行切碎(单字符识别差)。
      这里改为: 白色二值化 → 投影定位整行数字 bbox → 裁出 → use_det=False 整行识别。
      若整行无字, 退回整ROI检测; 再退反色。

用法:
    from score_ocr import read_score
    val = read_score(bgr_frame, debug=True)   # → int | None
"""
import os
import re
import sys

import cv2
import numpy as np

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_LIBS = os.path.join(BASE, "libs_ocr")
if os.path.isdir(_LIBS) and _LIBS not in sys.path:
    sys.path.insert(0, _LIBS)

# 分数 ROI: (x0, x1, y0, y1) 占屏比例
SCORE_ROI = (0.900, 0.985, 0.050, 0.096)
BRIGHT = 190                # 白字阈值
TARGET_W = 640.0            # 放大后目标宽度

_engine = None


def _get_engine():
    global _engine
    if _engine is None:
        from rapidocr_onnxruntime import RapidOCR
        _engine = RapidOCR()
    return _engine


def crop_score(bgr):
    h, w = bgr.shape[:2]
    x0, x1, y0, y1 = SCORE_ROI
    return bgr[int(h * y0):int(h * y1), int(w * x0):int(w * x1)]


def _to_int(texts):
    s = "".join(texts)
    s = re.sub(r"[^0-9]", "", s)
    return int(s) if s else None


def _locate_line(roi):
    """白色二值化 → 行/列投影 → 返回裁好的整行数字图(含逗号), 无字→None。"""
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    m = (gray > BRIGHT).astype(np.uint8)
    k = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, k)
    rows = m.sum(axis=1)
    ys = np.flatnonzero(rows > 0)
    if ys.size == 0:
        return None
    y0, y1 = int(ys[0]), int(ys[-1])
    sub = m[y0:y1 + 1, :]
    cols = sub.sum(axis=0)
    xs = np.flatnonzero(cols > 0)
    if xs.size == 0:
        return None
    x0, x1 = int(xs[0]), int(xs[-1])
    pad = 3
    x0 = max(0, x0 - pad); x1 = min(roi.shape[1], x1 + pad)
    return roi[y0:y1 + 1, x0:x1]


def _rec_line(line_bgr):
    """对整行数字图 use_det=False 识别, 返回文本列表。"""
    h, w = line_bgr.shape[:2]
    up = max(2, int(round(TARGET_W / max(w, 1))))
    big = cv2.resize(line_bgr, None, fx=up, fy=up, interpolation=cv2.INTER_CUBIC)
    out = _get_engine()(big, use_det=False)
    result = out[0] if isinstance(out, tuple) else out
    texts = []
    if result:
        for item in result:
            # use_det=False 返回形式可能是 [box, text, score] 或 [text, score]
            if isinstance(item, (list, tuple)):
                t = item[1] if len(item) >= 2 and isinstance(item[1], str) else (
                    item[0] if isinstance(item[0], str) else None)
            else:
                t = item
            if isinstance(t, str):
                texts.append(t)
    return texts


def read_score(bgr, debug=False):
    """读右上角分数。成功→int, 失败/无字→None。"""
    roi = crop_score(bgr)
    if roi.size == 0:
        return None
    # 1) 投影定位整行数字 → 正色识别
    line = _locate_line(roi)
    texts_a = _rec_line(line) if line is not None else []
    pick = texts_a
    n_pick = len(str(_to_int(texts_a))) if _to_int(texts_a) is not None else 0
    # 2) 正色失败才反色(黑字白底) — 省一半推理
    if n_pick == 0 and line is not None:
        gray = cv2.cvtColor(line, cv2.COLOR_BGR2GRAY)
        inv = cv2.cvtColor(255 - gray, cv2.COLOR_GRAY2BGR)
        texts_b = _rec_line(inv)
        n_b = len(str(_to_int(texts_b))) if _to_int(texts_b) is not None else 0
        if n_b > 0:
            pick, n_pick = texts_b, n_b
    # 3) 兜底: 整ROI默认检测
    if n_pick == 0:
        try:
            out = _get_engine()(roi)
            result = out[0] if isinstance(out, tuple) else out
            texts_c = [it[1] for it in (result or []) if len(it) >= 2]
            n_pick = len(str(_to_int(texts_c))) if _to_int(texts_c) is not None else 0
            if n_pick > 0:
                pick = texts_c
        except Exception:
            pass
    if debug:
        print("[score_ocr] line=%s 正=%s → %s"
              % ("有" if line is not None else "无", texts_a, _to_int(pick)))
    return _to_int(pick)


if __name__ == "__main__":
    def im_u(p):
        return cv2.imdecode(np.fromfile(p, dtype=np.uint8), cv2.IMREAD_COLOR)

    import glob
    files = sorted(glob.glob(os.path.join(r"D:\AI\临时文件\分数", "*.png")),
                   key=os.path.getmtime)
    if not files:
        print("无图")
        sys.exit(0)
    for f in files[-3:]:
        v = read_score(im_u(f), debug=True)
        print(f"{os.path.basename(f)} → {v}")
