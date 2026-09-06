# -*- coding: utf-8 -*-
"""
右上角分数 ROI 标定采集器 (Score ROI Capturer) — 击杀奖励 OCR 前哨
=================================================================
用途：跑一小段(默认15s)，把右上角"分数区域"原图+裁剪存盘，
     并打印该区域的二值化统计(亮像素比/字符块数/各块bbox宽高)，
     供人眼对照屏幕，迭代确定准确 ROI 与分数外观。

用法:
    python scripts/capture_score.py --seconds 20
参数:
    --region  "left,top,width,height" 游戏区域(默认全屏)
    --roi     "x0,y0,x1,y1"(屏幕百分比, 默认 "68,1,100,9")
    --out     输出目录(默认 D:\\AI\\临时文件\\score_samples)
    --seconds 采集秒数(默认 15)
    --every   每几秒抓一次(默认 0.8)
"""
import os
import time
import argparse

import cv2
import numpy as np
import mss

DEFAULT_OUT = r"D:\AI\临时文件\score_samples"


def save_img(path, img):
    """cv2.imwrite 中文路径静默失败 → imencode+二进制写(unicode 安全)。"""
    ok, buf = cv2.imencode(".png", img)
    if not ok:
        return False
    with open(path, "wb") as f:
        f.write(buf.tobytes())
    return True


def analyze(bgr):
    """对 ROI 做亮字二值化 + 形态学闭运算 + 字符块分割，返回 (统计文本, 标注图)。"""
    g = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    light = (g > 200).astype(np.uint8)
    k = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    light = cv2.morphologyEx(light, cv2.MORPH_CLOSE, k)
    n, lab, stats, cents = cv2.connectedComponentsWithStats(light, 8)
    h, w = bgr.shape[:2]
    lines = ["ROI=%dx%d 亮像素比=%.3f 块数=%d" % (w, h, float(light.mean()), n - 1)]
    annot = bgr.copy()
    for i in range(1, n):
        x, y, bw, bh, area = stats[i]
        if area < 8:
            continue
        lines.append("  块%d: x=%d y=%d w=%d h=%d area=%d 宽高比=%.2f"
                     % (i, x, y, bw, bh, area, bw / max(bh, 1)))
        cv2.rectangle(annot, (x, y), (x + bw, y + bh), (0, 255, 0), 1)
    return "\n".join(lines), annot


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--region", default=None, help='"left,top,width,height"，默认全屏')
    parser.add_argument("--roi", default="68,1,100,9", help='分数区百分比 "x0,y0,x1,y1"')
    parser.add_argument("--out", default=DEFAULT_OUT, help="输出目录")
    parser.add_argument("--seconds", type=float, default=15.0)
    parser.add_argument("--every", type=float, default=0.8)
    args = parser.parse_args()

    os.makedirs(args.out, exist_ok=True)
    r = [float(x) for x in args.roi.split(",")]

    with mss.mss() as sct:
        if args.region:
            l, t, w, h = [int(x) for x in args.region.split(",")]
            monitor = {"left": l, "top": t, "width": w, "height": h}
        else:
            monitor = sct.monitors[1]
            l, t, w, h = monitor["left"], monitor["top"], monitor["width"], monitor["height"]

        x0, y0 = int(w * r[0] / 100), int(h * r[1] / 100)
        x1, y1 = int(w * r[2] / 100), int(h * r[3] / 100)
        print("=" * 55)
        print("分数ROI标定采集器 | 区域 %dx%d @(%d,%d)" % (w, h, l, t))
        print("ROI 像素: (%d,%d)-(%d,%d)  = 屏幕百分比 %s" % (x0, y0, x1, y1, args.roi))
        print("输出: %s | 时长 %ss" % (args.out, args.seconds))
        print("=" * 55)

        t_end = time.time() + args.seconds
        count = 0
        while time.time() < t_end:
            t0 = time.time()
            img = np.array(sct.grab(monitor))
            img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
            roi = img[y0:y1, x0:x1]
            text, annot = analyze(roi)
            print("--- t=%.1fs ---" % (t_end - time.time()))
            print(text)
            ts = int(time.time() * 1000)
            if count % 3 == 0:                      # 每 ~2.4s 存一组
                save_img(os.path.join(args.out, "roi_%d.png" % ts), roi)
                save_img(os.path.join(args.out, "annot_%d.png" % ts), annot)
            count += 1
            dt = time.time() - t0
            if dt < args.every:
                time.sleep(args.every - dt)

    print("完成，样本已存 %s" % args.out)
    print("请把上面的块统计 + 屏幕肉眼对照结果发给我：ROI 框准了吗？数字是亮色吗？")


if __name__ == "__main__":
    main()
