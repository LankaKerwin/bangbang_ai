# -*- coding: utf-8 -*-
"""
从"葬身大海"结算截图中自动抠出红色大标题区域，存成死亡模板。
用法: python scripts/make_death_template.py
输出: assets/templates/death_da_zi.png （供 auto_restart 匹配死亡）
"""
import os
import glob
import cv2
import numpy as np

SRC_GLOB = r"D:\AI\临时文件\Bang Bang Barrage 2026_9_5 7_49_44.png"
OUT_DIR = r"D:\AI\AI train\bangbang_ai\assets\templates"


def imread_unicode(path):
    """cv2.imread 读不了中文路径 → 二进制读入再 imdecode"""
    with open(path, "rb") as f:
        data = np.frombuffer(f.read(), dtype=np.uint8)
    return cv2.imdecode(data, cv2.IMREAD_COLOR)


def main():
    matches = glob.glob(SRC_GLOB)
    if not matches:
        print("找不到死亡结算截图:", SRC_GLOB)
        return
    img = imread_unicode(matches[0])
    if img is None:
        print("读取失败")
        return
    h, w = img.shape[:2]
    b, g, r = cv2.split(img)
    red = ((r > 130) & (g < 110) & (b < 110)).astype(np.uint8) * 255

    # 只看上中部(标题区)，横 10%-90%
    y0, y1 = int(h * 0.08), int(h * 0.60)
    x0, x1 = int(w * 0.10), int(w * 0.90)
    zone = red[y0:y1, x0:x1].copy()
    k = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15))
    zone = cv2.morphologyEx(zone, cv2.MORPH_CLOSE, k)
    n, lab, stats, _ = cv2.connectedComponentsWithStats(zone, 8)

    comps = []
    for i in range(1, n):
        x, y, bw, bh, area = stats[i]
        if bh > h * 0.03 and bw > w * 0.02 and area > 5000:
            comps.append((x0 + x, y0 + y, bw, bh, area))
    comps.sort(key=lambda c: -c[4])
    print("红色大块(top5):")
    for c in comps[:5]:
        print("  x=%d y=%d w=%d h=%d area=%d  pct=(%d%%,%d%%)"
              % (c[0], c[1], c[2], c[3], c[4], round(c[0] / w * 100), round(c[1] / h * 100)))

    if not comps:
        print("没找到红色大字")
        return

    top = comps[:3]
    X = min(c[0] for c in top)
    Y = min(c[1] for c in top)
    X2 = max(c[0] + c[2] for c in top)
    Y2 = max(c[1] + c[3] for c in top)
    pad = 10
    X, Y = max(0, X - pad), max(0, Y - pad)
    X2, Y2 = min(w, X2 + pad), min(h, Y2 + pad)

    os.makedirs(OUT_DIR, exist_ok=True)
    crop = img[Y:Y2, X:X2]
    outp = os.path.join(OUT_DIR, "death_da_zi.png")
    ok, buf = cv2.imencode(".png", crop)
    if ok:
        with open(outp, "wb") as f:
            f.write(buf.tobytes())
        print("已保存模板:", outp)
        print("裁剪区域 pct: x %d-%d%%, y %d-%d%%, 模板尺寸 %dx%d"
              % (round(X / w * 100), round(X2 / w * 100),
                 round(Y / h * 100), round(Y2 / h * 100), crop.shape[1], crop.shape[0]))


if __name__ == "__main__":
    main()
