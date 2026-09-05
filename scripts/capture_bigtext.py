"""
大字自动捕获器 (BigText Auto-Capture) — M0b 感知素材采集工具
==============================================================
用途：游玩时自动捕捉屏幕中央【大号公告文字】（如 特殊事件名 /
      Boss名+即将出现 / 恢复正常 / 奖励时间 / 葬身大海），存图供
      L3 大字模板建库（模板匹配用，不用 OCR）。

原理：
- 以 ~8fps 扫描屏幕中央"公告带"（默认 x 12%~88%、y 22%~72%）
- 检测"大面积 亮色/红色 大字形"的 上升沿（从无到有）→ 立即截一张
- 冷却 6s + 内容指纹去重，避免重复存图
- 停止：按 Ctrl+Shift+X，或直接关闭命令行窗口

用法：
    python scripts/capture_bigtext.py
    python scripts/capture_bigtext.py --out "D:\临时文件\大字样本" --fps 10

参数：
    --region   "left,top,width,height" 游戏区域，默认全屏
    --zone     "x0,y0,x1,y1"(百分比) 中央公告带，默认 "12,22,88,72"
    --out      输出目录，默认 D:\AI\临时文件\大字样本
    --fps      扫描频率，默认 8
    --cooldown 冷却秒数，默认 6
    --min_ratio 大字形像素占比阈值(触发)，默认 0.008（越高越不易误触发）
"""
import os
import time
import hashlib
import argparse

import cv2
import numpy as np
import mss

DEFAULT_OUT = r"D:\AI\临时文件\大字样本"


def _dhash(gray_zone, size=16):
    """内容指纹：缩小灰度图 → 相邻像素差哈希，用于去重"""
    small = cv2.resize(gray_zone, (size + 1, size))
    diff = small[:, 1:] > small[:, :-1]
    return hashlib.md5(diff.tobytes()).hexdigest()[:16]


def text_score(bgr, zone):
    """返回中央公告带的"大字形信号"。返回 (亮/红像素占比, 大块数量)。"""
    x0, y0, x1, y1 = zone
    g = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    gg = g[y0:y1, x0:x1]
    b = bgr[y0:y1, x0:x1, 0]
    r = bgr[y0:y1, x0:x1, 2]

    # 亮字(白/黄等) + 大红字(如 葬身大海/毒 的红色大字)
    light = (gg > 205).astype(np.uint8)
    red = ((r > 170) & (gg < 120)).astype(np.uint8)
    mask = np.clip(light + red, 0, 1).astype(np.uint8)

    # 形态学把相邻笔画连成"字块"
    k = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 9))
    closed = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k)

    n, lab, stats, _ = cv2.connectedComponentsWithStats(closed, 8)
    zarea = (x1 - x0) * (y1 - y0)
    big_count = 0
    for i in range(1, n):
        if stats[i, 4] > zarea * 0.004:   # 大于 0.4% 区域的块算"大字形"
            big_count += 1
    ratio = float(mask.sum()) / zarea
    return ratio, big_count


def save_img(path, img):
    """cv2.imwrite 对含中文的路径会【静默失败】(返回False) → 用 imencode+二进制写，unicode 安全。"""
    ok, buf = cv2.imencode(".png", img)
    if not ok:
        return False
    with open(path, "wb") as f:
        f.write(buf.tobytes())
    return True


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--region", default=None, help='"left,top,width,height"，默认全屏')
    parser.add_argument("--zone", default="12,22,88,72", help='公告带百分比 "x0,y0,x1,y1"')
    parser.add_argument("--out", default=DEFAULT_OUT, help="输出目录")
    parser.add_argument("--fps", type=float, default=8.0)
    parser.add_argument("--cooldown", type=float, default=6.0)
    parser.add_argument("--min_ratio", type=float, default=0.008)
    args = parser.parse_args()

    os.makedirs(args.out, exist_ok=True)
    z = [int(float(x)) for x in args.zone.split(",")]

    with mss.mss() as sct:
        if args.region:
            l, t, w, h = [int(x) for x in args.region.split(",")]
            monitor = {"left": l, "top": t, "width": w, "height": h}
        else:
            monitor = sct.monitors[1]
            l, t, w, h = monitor["left"], monitor["top"], monitor["width"], monitor["height"]

        print("=" * 55)
        print("大字自动捕获器已启动")
        print(f"区域: {w}x{h} @({l},{t}) | 公告带: {args.zone}")
        print(f"输出: {args.out}")
        print(f"冷却 {args.cooldown}s | 阈值 {args.min_ratio}")
        print("  Ctrl+Shift+X 退出；也可直接关闭窗口")
        print("=" * 55)

        import keyboard  # 延迟导入，装不上也不影响主功能
        running = {"v": True}
        keyboard.add_hotkey("ctrl+shift+x", lambda: running.update(v=False))

        interval = 1.0 / args.fps
        last_cap = 0.0
        prev_active = False
        prev_digest = None
        count = 0

        while running["v"]:
            t0 = time.time()
            img = np.array(sct.grab(monitor))
            img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)

            hh, ww = img.shape[:2]
            zone = (int(ww * z[0] / 100), int(hh * z[1] / 100),
                    int(ww * z[2] / 100), int(hh * z[3] / 100))
            ratio, nbig = text_score(img, zone)
            active = ratio > args.min_ratio and nbig >= 1

            # 上升沿 = 大字形"刚出现"的瞬间
            if active and not prev_active and (time.time() - last_cap) > args.cooldown:
                gzone = cv2.cvtColor(img[zone[1]:zone[3], zone[0]:zone[2]],
                                     cv2.COLOR_BGR2GRAY)
                digest = _dhash(gzone)
                if digest != prev_digest:
                    fname = f"bigtext_{int(time.time()*1000)}.png"
                    ok1 = save_img(os.path.join(args.out, fname), img)
                    # 顺带存一张裁剪的公告带小图，方便人快速浏览
                    ok2 = save_img(os.path.join(args.out, "crop_" + fname),
                                   img[zone[1]:zone[3], zone[0]:zone[2]])
                    if ok1 or ok2:
                        count += 1
                        print(f"  [{count}] 捕获 {fname}  (亮比 {ratio:.3f}, 字块 {nbig})"
                              + ("" if ok1 else "  [警告:全图保存失败]"))
                    else:
                        print(f"  [警告] {fname} 保存失败(检查输出目录可写)")
                    last_cap = time.time()
                    prev_digest = digest
            prev_active = active

            dt = time.time() - t0
            if dt < interval:
                time.sleep(interval - dt)

    print(f"\n完成，共捕获 {count} 张 → {args.out}")
    print("下一步：按事件名把 crop_*.png 归类命名，作为 L3 大字模板库")


if __name__ == "__main__":
    main()
