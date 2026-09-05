"""
枪豆人 训练驱动 (M1 雏形 v2) — AI 自主游玩 + 死亡自重启 + 商店分区
====================================================================
单进程状态机，共享一个虚拟 Xbox360 手柄(vgamepad)：
  [战斗中]  策略(现有YOLO: 最近敌→右摇杆【满杆】瞄准 → RT开火)
  [商店]    检测到顶部 SHOP 招牌 → 摇杆/扳机清零、冻结策略（后续接商店购买策略）
  [死亡重开] 葬身大海(红字+血量空心) → 摇杆清零 → 长按A跳过加载 → 双按A开新局
之后回到 [战斗中]。策略可插拔：M0b 后用脚本教练/PPO 替换 policy()。

用法：
    python scripts/train_driver.py
    python scripts/train_driver.py --region "0,0,1280,720" --dry --show
热键：F10 开始/暂停 ｜ F8 自动重开开关 ｜ F9 手动重开 ｜ Ctrl+Shift+X 退出
"""
import os
import time
import argparse

import numpy as np
import cv2
import mss
import vgamepad as vg
from ultralytics import YOLO
from auto_restart import _norm_mask_red, _count_solid_hearts  # 复用死亡检测
from perception import state_from_frame, enemy_in_range        # M0b 感知

# ===== 可调 =====
BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE, "models", "yolo", "bb_yolov8s", "weights", "best.pt")
SHOP_TPL = os.path.join(BASE, "assets", "templates", "shop.png")
CONF = 0.25
PERSIST_FRAMES = 10      # (保留，诊断用；主死亡判断改用心跳血量)
HEART_DEAD_FRAMES = 6    # 血量(实心心)连续为0的帧数 → 判死(颜色无关，红/灰都算)
MIN_RED_RATIO = 0.010
SHOP_PERSIST = 3
SHOP_SCORE = 0.60      # 收紧：防海面/背景误判成 SHOP
SHOP_SKIP_INTERVAL = 2.5   # 商店自动退出动作的最小间隔(s)
HOLD_SKIP_SEC = 2.5
WAIT_AFTER_SKIP = 1.2
A_PRESS_SEC = 0.15
SECOND_PRESS_GAP = 0.8
AFTER_RESTART_WAIT = 2.5
COOLDOWN_SEC = 8.0
FIRE_KEEP_SEC = 0.8      # 看到敌人后至少维持"可开火"这么久(去抖)
FIRE_PERIOD_SEC = 0.4    # 半自动：每隔多久扣一次扳机(RT)
RT_PULSE_SEC = 0.08      # 每次扣扳机按住时长
PLAYER_CLS = 0
# =====================


def imread_u(path):
    with open(path, "rb") as f:
        return cv2.imdecode(np.frombuffer(f.read(), np.uint8), cv2.IMREAD_COLOR)


def reset_controls(gp):
    """左右摇杆+左右扳机全部归中/松开，避免菜单被摇杆带动。"""
    gp.left_joystick_float(0.0, 0.0)
    gp.right_joystick_float(0.0, 0.0)
    gp.left_trigger_float(0.0)
    gp.right_trigger_float(0.0)
    gp.update()


def skip_shop(gp):
    """默认不买商店：连按 B(返回) 和 A(确认) 顺序关掉商店，退回游戏。"""
    for _ in range(2):
        gp.press_button(vg.XUSB_BUTTON.XUSB_GAMEPAD_B); gp.update(); time.sleep(0.20)
        gp.release_button(vg.XUSB_BUTTON.XUSB_GAMEPAD_B); gp.update(); time.sleep(0.25)
        gp.press_button(vg.XUSB_BUTTON.XUSB_GAMEPAD_A); gp.update(); time.sleep(0.12)
        gp.release_button(vg.XUSB_BUTTON.XUSB_GAMEPAD_A); gp.update(); time.sleep(0.25)
    reset_controls(gp)


def detect_shop(frame, tpl):
    """顶部中央匹配 SHOP 招牌。返回是否在商店(带持续帧判定在外层)。"""
    if tpl is None:
        return 0.0
    h, w = frame.shape[:2]
    y0, y1 = int(h * 0.00), int(h * 0.16)
    x0, x1 = int(w * 0.20), int(w * 0.80)
    roi = cv2.cvtColor(frame[y0:y1, x0:x1], cv2.COLOR_BGR2GRAY)
    t = cv2.cvtColor(tpl, cv2.COLOR_BGR2GRAY)
    if roi.shape[0] < t.shape[0] or roi.shape[1] < t.shape[1]:
        return 0.0
    res = cv2.matchTemplate(roi, t, cv2.TM_CCOEFF_NORMED)
    _, mx, _, _ = cv2.minMaxLoc(res)
    return float(mx)


def find_player(res, frame):
    h, w = frame.shape[:2]
    boxes = res[0].boxes
    if boxes is not None and len(boxes) > 0:
        for bb in boxes:
            if int(bb.cls[0]) == PLAYER_CLS:
                x1, y1, x2, y2 = [float(v) for v in bb.xyxy[0].tolist()]
                return ((x1 + x2) / 2, (y1 + y2) / 2)
    return (w / 2, h / 2)


def policy(res, frame, gp, pbox, prev_state=None):
    """用已预测 res + perception 状态判定。
    返回 (state, 可开火?, 敌人数)。只负责 右摇杆满杆瞄准最近敌人；
    开火由主循环按节奏扣 RT（框到且有弹才算可开火）。"""
    st = state_from_frame(frame, pbox, prev_state)
    boxes = res[0].boxes if res is not None else None
    n_enemy = 0
    if boxes is None or len(boxes) == 0:
        reset_controls(gp)
        return st, False, n_enemy
    px, py = pbox
    best, best_d = None, 1e18
    for bb in boxes:
        if int(bb.cls[0]) == PLAYER_CLS:
            continue
        n_enemy += 1
        x1, y1, x2, y2 = [float(v) for v in bb.xyxy[0].tolist()]
        cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
        d = (cx - px) ** 2 + (cy - py) ** 2
        if d < best_d:
            best_d, best = d, (cx, cy)
    if best is None:
        reset_controls(gp)
        return st, False, n_enemy
    dx, dy = best[0] - px, best[1] - py
    norm = (dx * dx + dy * dy) ** 0.5
    if norm <= 1:
        reset_controls(gp)
        return st, False, n_enemy
    gp.right_joystick_float(x_value_float=float(np.clip(dx / norm, -1, 1)),
                            y_value_float=float(np.clip(-dy / norm, -1, 1)))
    inr = enemy_in_range(st, best, pbox)
    can_fire = bool(inr) and st.get("ammo") is not None and st["ammo"] >= 1
    gp.update()
    return st, can_fire, n_enemy


def dead_now(frame, z_center, z_hp):
    """快速判断是否仍在死亡/结算画面（红字 + 血量空心）。"""
    red_ratio = _norm_mask_red(frame, z_center)
    hearts = _count_solid_hearts(frame, z_hp)
    return red_ratio > MIN_RED_RATIO and hearts <= 0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--region", default=None)
    parser.add_argument("--model", default=MODEL_PATH)
    parser.add_argument("--dry", action="store_true")
    parser.add_argument("--show", action="store_true")
    parser.add_argument("--verbose", action="store_true", help="打印检测/开火状态")
    parser.add_argument("--no_shop", action="store_true", help="禁用商店检测(先把商店冻结关掉测战斗)")
    args = parser.parse_args()

    model = YOLO(args.model)
    shop_tpl = imread_u(SHOP_TPL) if os.path.exists(SHOP_TPL) else None
    import keyboard
    gp = vg.VX360Gamepad()
    reset_controls(gp)
    running = {"v": True, "play": False, "arm": True}
    keyboard.add_hotkey("f10", lambda: running.update(play=not running["play"]))
    keyboard.add_hotkey("f8", lambda: running.update(arm=not running["arm"]))
    keyboard.add_hotkey("ctrl+shift+x", lambda: running.update(v=False))

    if args.show:
        cv2.namedWindow("train", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("train", 480, 270)
        try:
            cv2.setWindowProperty("train", cv2.WND_PROP_TOPMOST, 1)
        except Exception:
            pass

    with mss.mss() as sct:
        if args.region:
            l, t, w, h = [int(x) for x in args.region.split(",")]
            monitor = {"left": l, "top": t, "width": w, "height": h}
        else:
            monitor = sct.monitors[1]
            l, t, w, h = monitor["left"], monitor["top"], monitor["width"], monitor["height"]
        z_center = (int(w * 0.06), int(h * 0.16), int(w * 0.94), int(h * 0.56))
        z_hp = (int(w * 0.015), int(h * 0.015), int(w * 0.20), int(h * 0.085))

        def grab():
            f = np.array(sct.grab(monitor))
            return cv2.cvtColor(f, cv2.COLOR_BGRA2BGR)

        def do_restart():
            """长按A跳过加载 → 循环验证：按A直到进入新局。带每秒诊断。"""
            reset_controls(gp)
            time.sleep(0.3)
            print("  [重开] 长按A跳过加载(%ss)..." % HOLD_SKIP_SEC)
            gp.press_button(vg.XUSB_BUTTON.XUSB_GAMEPAD_A); gp.update()
            time.sleep(HOLD_SKIP_SEC)
            gp.release_button(vg.XUSB_BUTTON.XUSB_GAMEPAD_A); gp.update()
            for i in range(14):
                time.sleep(1.0)
                fr = grab()
                rd = _norm_mask_red(fr, z_center)
                hs = _count_solid_hearts(fr, z_hp)
                still_dead = hs <= 0   # 血量还没回来 = 还在结算/结果界面(颜色无关)
                print(f"    t{i}: red={rd:.3f} hearts={hs} 仍死亡={still_dead}")
                if not still_dead:
                    print("  ✓ 新一局已开始")
                    return True
                gp.press_button(vg.XUSB_BUTTON.XUSB_GAMEPAD_A); gp.update()
                time.sleep(A_PRESS_SEC)
                gp.release_button(vg.XUSB_BUTTON.XUSB_GAMEPAD_A); gp.update()
            print("  ⚠ 多次补按仍未开新局（可能需人工）")
            return False

        keyboard.add_hotkey("f9", do_restart)

        print("=" * 55)
        print("训练驱动(M1 v2)  F10开始/暂停  F8自动重开  F9手动  Ctrl+Shift+X退出")
        print(f"状态: 战斗(最近敌满杆瞄准+RT) / 商店(冻结) / 死亡重开(长按A+双按A)"
              + ("  [DRY]" if args.dry else ""))
        print("=" * 55)

        interval = 0.1
        red_run = 0
        boat_run = 0
        shop_run = 0
        in_shop = False
        deaths = 0
        last_act = 0.0
        last_hit = 0.0
        last_shot = 0.0
        last_log = 0.0
        last_shop_act = 0.0
        prev_play = running["play"]
        prev_state = None
        while running["v"]:
            t0 = time.time()
            frame = np.array(sct.grab(monitor))
            frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)

            red_ratio = _norm_mask_red(frame, z_center)
            hearts = _count_solid_hearts(frame, z_hp)
            red_run = red_run + 1 if red_ratio > MIN_RED_RATIO else 0

            # 死亡判定：仅播放中，玩家船连续消失(结算/菜单/加载) → 判死（颜色无关，最可靠）
            dead = False
            res = None
            if running["play"] and not args.dry:
                res = model.predict(frame, imgsz=640, conf=CONF, verbose=False)
                boat = False
                bx = res[0].boxes
                if bx is not None and len(bx) > 0:
                    boat = any(int(b.cls[0]) == PLAYER_CLS for b in bx)
                boat_run = boat_run + 1 if not boat else 0
                dead = boat_run >= HEART_DEAD_FRAMES
            else:
                boat_run = 0

            shop_score = 0.0
            if not args.no_shop and shop_tpl is not None:
                shop_score = detect_shop(frame, shop_tpl)
                shop_run = shop_run + 1 if shop_score > SHOP_SCORE else 0
                in_shop = shop_run >= SHOP_PERSIST
            else:
                in_shop = False

            if running["play"] != prev_play:
                if running["play"]:
                    print("[开始] AI 接管")
                else:
                    print("[暂停] 摇杆归零")
                    reset_controls(gp)
                prev_play = running["play"]

            if dead and running["arm"] and (time.time() - last_act) > COOLDOWN_SEC:
                deaths += 1
                print(f"[死亡] #{deaths} red={red_ratio:.3f} hearts={hearts} → 摇杆清零+自动重开")
                if not args.dry:
                    do_restart()
                last_act = time.time()
                boat_run = 0
                continue

            if in_shop and not dead:
                if not args.dry and (time.time() - last_shop_act) > SHOP_SKIP_INTERVAL:
                    print("[商店] 检测到商店 → 自动退出(连按B+A)，不购买")
                    skip_shop(gp)
                    last_shop_act = time.time()
                else:
                    reset_controls(gp)  # 商店期先归零，等退出后恢复战斗
                if args.show:
                    pass
            elif running["play"] and not dead and not args.dry:
                pbox = find_player(res, frame)
                st, can_fire, n_enemy = policy(res, frame, gp, pbox, prev_state)
                prev_state = st
                if can_fire:
                    last_hit = time.time()
                # 半自动：可开火(框到且有弹)才按节奏扣扳机
                if can_fire and (time.time() - last_shot) >= FIRE_PERIOD_SEC:
                    gp.right_trigger_float(1.0); gp.update(); time.sleep(RT_PULSE_SEC)
                    gp.right_trigger_float(0.0); gp.update()
                    last_shot = time.time()
                if args.verbose and (time.time() - last_log) > 1.0:
                    last_log = time.time()
                    aim = round(st["aim_deg"], 1) if st["aim_deg"] is not None else None
                    print(f"  [DBG] 敌={n_enemy} 弹={st['ammo']} aim={aim} "
                          f"半角={st['half']} 开火={can_fire} 商店={in_shop}")
            else:
                reset_controls(gp)  # 暂停态保持摇杆/扳机归零

            if args.show:
                res = model.predict(frame, imgsz=640, conf=CONF, verbose=False)
                ann = res[0].plot()
                cv2.putText(ann, f"deaths:{deaths} shop:{in_shop}", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                hh, ww = ann.shape[:2]
                cv2.imshow("train", cv2.resize(ann, (480, int(hh * 480 / ww))))
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

            dt = time.time() - t0
            if dt < interval:
                time.sleep(interval - dt)

    reset_controls(gp)
    print(f"\n退出。共死亡 {deaths} 次")
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
