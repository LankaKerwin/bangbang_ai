# -*- coding: utf-8 -*-
"""临时：分析红果/蓝果/钻石特写，提取主色与尺寸。"""
import glob, os
import cv2, numpy as np


def rd(p):
    with open(p, "rb") as f:
        return cv2.imdecode(np.frombuffer(f.read(), np.uint8), cv2.IMREAD_COLOR)


def color_stats(img):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    H, S, V = hsv[:, :, 0].astype(int), hsv[:, :, 1].astype(int), hsv[:, :, 2].astype(int)
    # 背景=最普遍的色(海域偏蓝) → 用饱和+色相远离背景来挑目标
    # 简化：取 饱和度 较亮 且 不是最常见色 的像素
    sat_colored = (S > 90) & (V > 90)
    if sat_colored.sum() < 30:
        return None
    hues = H[sat_colored]
    hist = np.bincount(hues, minlength=180)
    # 找最大的连续色带(忽略背景蓝 90-115 附近也可以)
    order = np.argsort(hist)[::-1]
    top = int(order[0])
    m = (H == top) & sat_colored
    mean_s = float(S[m].mean()); mean_v = float(V[m].mean())
    ys, xs = np.where(m)
    bw = xs.max() - xs.min() + 1; bh = ys.max() - ys.min() + 1
    return {"h": top, "s": round(mean_s, 1), "v": round(mean_v, 1),
            "px": int(m.sum()), "w": int(bw), "h": int(bh),
            "overall": (hsv[:, :, 0].shape, np.percentile(H[sat_colored], [10, 50, 90]).round(0).tolist())}


def main():
    for name in ("红果.png", "蓝果.png", "钻石.png"):
        p = os.path.join(r"D:\AI\临时文件\感知标定", name)
        if not os.path.exists(p):
            continue
        img = rd(p)
        st = color_stats(img)
        print(name, "→", st)


if __name__ == "__main__":
    main()
