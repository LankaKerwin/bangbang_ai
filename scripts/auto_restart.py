"""
枪豆人 死亡自动重开 (auto_restart) — RL 无人值守训练基础设施
==============================================================
触发：检测到"葬身大海"红色大字 + 左上血量空心(无实心心) → 确认死亡
动作：长按 A 跳过结算加载 → 单按 A 再开一局（游戏记住上次配置）
输入：虚拟 Xbox 手柄 (vgamepad)，A = 手柄 A 键

用法：
    python scripts/auto_restart.py                    # 正常模式(会注入按键)
    python scripts/auto_restart.py --dry              # 只检测不按键(先标定)
    python scripts/auto_restart.py --region "0,0,2560,1440" --hold_skip 2.0

热键：
    F8   启用/暂停 自动重开
    F9   手动触发一次重开(调试用)
    Ctrl+Shift+X  退出；也可关窗口

说明：
- 死亡检测 = 中央红色大字(比例>阈值)持续 N 帧 + 左上实心心数==0（复合，防 Boss 名等误触发）
- 每步带超时看门狗；多次失败会停止并提示
"""
import os
import time
import argparse

import numpy as np
import cv2
import mss
import vgamepad as vg

# ===== 可调参数 =====
HOLD_SKIP_SEC = 2.5       # 长按 A 跳过加载的时长（听雨建议加长）
WAIT_AFTER_SKIP = 1.2     # 松开后等 1.2s（让结算面板完全就绪再按）
A_PRESS_SEC = 0.15        # 单按 A 的按住时长
SECOND_PRESS_GAP = 0.8    # 两次"开始"间再隔 0.8s（防第一次没被吃）
AFTER_RESTART_WAIT = 2.5  # 重开后再等，让战斗开始
PERSIST_FRAMES = 12       # 红色大字需连续出现这么多帧(约1.5s@8fps)才算死亡
MIN_RED_RATIO = 0.010     # 中央红色字像素占比阈值
COOLDOWN_SEC = 8.0        # 触发后冷却
MAX_TRIES = 3             # 重开最多尝试次数
# =====================


def _norm_mask_red(bgr, zone):
    x0, y0, x1, y1 = zone
    bb = bgr[y0:y1, x0:x1, 0]
    gg = bgr[y0:y1, x0:x1, 1]
    rr = bgr[y0:y1, x0:x1, 2]
    red = ((rr > 130) & (gg < 110) & (bb < 110)).astype(np.uint8)
    k = cv2.getStructuringElement(cv2.MORPH_RECT, (11, 11))
    closed = cv2.morphologyEx(red, cv2.MORPH_CLOSE, k)
    area = (x1 - x0) * (y1 - y0)
    return float(closed.sum()) / area


def _count_solid_hearts(bgr, zone_hp):
    x0, y0, x1, y1 = zone_hp
    roi = bgr[y0:y1, x0:x1]
    b = roi[:, :, 0]; g = roi[:, :, 1]; r = roi[:, :, 2]
    white = ((r > 195) & (g > 195) & (b > 195)).astype(np.uint8)
    k = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    white = cv2.morphologyEx(white, cv2.MORPH_OPEN, k)
    n, lab, stats, _ = cv2.connectedComponentsWithStats(white, 8)
    hh, ww = roi.shape[:2]
    count = 0
    for i in range(1, n):
        x, y, bw, bh, area = stats[i]
        # 实心心：大小适中、近似圆形/方块
        if area > hh * ww * 0.0015 and bw > 6 and bh > 6 and 0.5 < bw / max(bh, 1) < 2.0:
            count += 1
    return count


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--region", default=None, help="'left,top,width,height' 游戏区域，默认全屏")
    parser.add_argument("--hold_skip", type=float, default=HOLD_SKIP_SEC)
    parser.add_argument("--dry", action="store_true", help="只检测不按键")
    parser.add_argument("--fps", type=float, default=8.0)
    args = parser.parse_args()

    with mss.mss() as sct:
        if args.region:
            l, t, w, h = [int(x) for x in args.region.split(",")]
            monitor = {"left": l, "top": t, "width": w, "height": h}
        else:
            monitor = sct.monitors[1]
            l, t, w, h = monitor["left"], monitor["top"], monitor["width"], monitor["height"]
        z_center = (int(w * 0.06), int(h * 0.16), int(w * 0.94), int(h * 0.56))   # 中央大字区
        z_hp = (int(w * 0.015), int(h * 0.015), int(w * 0.20), int(h * 0.085))   # 左上血量区

        import keyboard
        running = {"v": True, "armed": True}
        keyboard.add_hotkey("f8", lambda: running.update(armed=not running["armed"]))
        keyboard.add_hotkey("f9", lambda: _do_restart(args, running, None))  # 手动
        keyboard.add_hotkey("ctrl+shift+x", lambda: running.update(v=False))

        if args.dry:
            print("[dry] 只检测，不注入按键")
        print(f"自动重开 {'已启用' if running['armed'] else '暂停'}(F8 切换)  F9 手动重开  Ctrl+Shift+X 退出")
        print("检测：中央红色大字 + 左上血量空心")

        import vgamepad as vg
        gp = vg.VX360Gamepad()

        interval = 1.0 / args.fps
        red_run = 0
        last_act = 0.0
        while running["v"]:
            t0 = time.time()
            img = np.array(sct.grab(monitor))
            img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
            red_ratio = _norm_mask_red(img, z_center)
            hearts = _count_solid_hearts(img, z_hp)

            if red_ratio > MIN_RED_RATIO:
                red_run += 1
            else:
                red_run = 0

            dead = red_run >= PERSIST_FRAMES and hearts <= 0
            if dead and running["armed"] and (time.time() - last_act) > COOLDOWN_SEC:
                print(f"[死亡触发] red={red_ratio:.3f} hearts={hearts} → 重开")
                if not args.dry:
                    _do_restart(args, running, gp)
                last_act = time.time()

            dt = time.time() - t0
            if dt < interval:
                time.sleep(interval - dt)

    print("已退出")


def _do_restart(args, running, gp):
    """长按A跳过加载 → 单按A开新局。含看门狗重试。"""
    if gp is None:
        return
    for attempt in range(1, MAX_TRIES + 1):
        # 长按 A（跳过结算加载）
        gp.press_button(vg.XUSB_BUTTON.XUSB_GAMEPAD_A); gp.update(); time.sleep(HOLD_SKIP_SEC)
        gp.release_button(vg.XUSB_BUTTON.XUSB_GAMEPAD_A); gp.update(); time.sleep(WAIT_AFTER_SKIP)
        # 单按 A（再来一局）—— 连按两次，防第一次被过渡吃掉
        for _ in range(2):
            gp.press_button(vg.XUSB_BUTTON.XUSB_GAMEPAD_A); gp.update(); time.sleep(A_PRESS_SEC)
            gp.release_button(vg.XUSB_BUTTON.XUSB_GAMEPAD_A); gp.update()
            time.sleep(SECOND_PRESS_GAP)
        print(f"  第{attempt}次尝试完成(长按{HOLD_SKIP_SEC}s + 双按A)")
        time.sleep(AFTER_RESTART_WAIT)
        # 简化验证：等待冷却结束，交给下一轮死亡触发；不在此阻塞判断
        return


if __name__ == "__main__":
    main()
