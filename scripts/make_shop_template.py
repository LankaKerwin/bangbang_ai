# -*- coding: utf-8 -*-
"""从商店截图抠出顶部 SHOP 招牌区域作为模板，供训练驱动识别商店状态。"""
import glob
import os
import cv2
import numpy as np


def imread_u(path):
    with open(path, "rb") as f:
        return cv2.imdecode(np.frombuffer(f.read(), np.uint8), cv2.IMREAD_COLOR)


def main():
    p = glob.glob(r"D:\AI\临时文件\Bang Bang Barrage 2026_9_5 7_43_30.png")[0]  # 商店截图
    img = imread_u(p)
    h, w = img.shape[:2]
    # 顶部中央 SHOP 招牌区（含蓝色船身 + SHOP 字），保守框取
    x0, x1 = int(w * 0.36), int(w * 0.64)
    y0, y1 = int(h * 0.02), int(h * 0.14)
    crop = img[y0:y1, x0:x1]
    out_dir = r"D:\AI\AI train\bangbang_ai\assets\templates"
    os.makedirs(out_dir, exist_ok=True)
    outp = os.path.join(out_dir, "shop.png")
    ok, buf = cv2.imencode(".png", crop)
    if ok:
        with open(outp, "wb") as f:
            f.write(buf.tobytes())
        print("shop template saved:", outp, crop.shape, "region pct=(%d-%d%%,%d-%d%%)"
              % (round(x0 / w * 100), round(x1 / w * 100), round(y0 / h * 100), round(y1 / h * 100)))


if __name__ == "__main__":
    main()
