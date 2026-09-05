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
import math
import argparse

import numpy as np
import cv2
import mss
import vgamepad as vg
from ultralytics import YOLO
from auto_restart import _norm_mask_red, _count_solid_hearts  # 复用死亡检测
from perception import state_from_frame, enemy_in_range, m0b_state  # M0b 感知
import bb_classes as B

# 类别分组(电锯=环境物默认不打; 幽灵=需持续照射)
SAW_IDS = {B.TOKEN_TO_ID[t] for t in B.TOKEN_TO_ID if t.startswith("saw_")}
GHOST_IDS = {B.TOKEN_TO_ID[t] for t in B.TOKEN_TO_ID if t.startswith("ghost_")}
HAT_GHOST_IDS = {B.TOKEN_TO_ID["ghost_hat"]}   # 帽子幽灵：先打帽才能照死
BUL_GHOST_DWELL = True   # 幽灵规则：锁定照射、不扣扳机
GHOST_DWELL_SEC = 2.0    # 幽灵持续照射时长(实测≈1.5s，保守2.0s)
EVENT_SAW_AREA_FRAC = 0.008   # 电锯框面积 > 帧面积×此比例 → 视为"事件电锯"(大,打不坏)
RELOAD_SEC = 2.5              # 内部弹药估计：打空后装填耗时估计(待实测微调)

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


def _aim_at(gp, pbox, target):
    """右摇杆满杆指向 target。"""
    px, py = pbox
    dx, dy = target[0] - px, target[1] - py
    norm = (dx * dx + dy * dy) ** 0.5
    if norm <= 1:
        reset_controls(gp)
        return False
    gp.right_joystick_float(x_value_float=float(np.clip(dx / norm, -1, 1)),
                            y_value_float=float(np.clip(-dy / norm, -1, 1)))
    gp.update()
    return True


def policy(res, frame, gp, pbox, prev_state=None, lock=None):
    """规则化目标选择 v3：
    1) 射程内有普通敌人 → 先打(不因幽灵打断) 'shoot'/'wait'
    2) 幽灵(射程内无敌人时)：帽子幽灵先开火打帽 'hat'；普通幽灵锁定照射 'ghost'
    3) 大电锯(事件) 默认不打；进射程或无可打才打 'saw'
    返回 (st, inr(True/False/None), mode, n)。
    inr=None = 感知读不到范围(如暴风雨)→ 不算"确定在范围外"，主循环允许开火。"""
    st = state_from_frame(frame, pbox, prev_state)
    boxes = res[0].boxes if res is not None else None
    if boxes is None or len(boxes) == 0:
        if lock is not None:
            lock.pop("gc", None)
        reset_controls(gp)
        return st, None, "idle", 0

    hh, ww = frame.shape[:2]
    farea = hh * ww
    regs, ghosts, saw_big = [], [], []
    for bb in boxes:
        cid = int(bb.cls[0])
        if cid == PLAYER_CLS:
            continue
        x1, y1, x2, y2 = [float(v) for v in bb.xyxy[0].tolist()]
        c = ((x1 + x2) / 2, (y1 + y2) / 2)
        area = (x2 - x1) * (y2 - y1)
        if cid in GHOST_IDS:
            ghosts.append((c, cid))
        elif cid in SAW_IDS:
            if area > EVENT_SAW_AREA_FRAC * farea:
                saw_big.append(c)
            else:
                regs.append(c)
        else:
            regs.append(c)

    def nearest_center(lst):
        return min(lst, key=lambda cc: (cc[0] - pbox[0]) ** 2 + (cc[1] - pbox[1]) ** 2) if lst else None

    def inr_of(e):
        return enemy_in_range(st, e, pbox)

    # 1) 射程内的普通敌人(含未知范围None) 优先——避免幽灵打断交火
    reg_in = [r for r in regs if inr_of(r) is not False]
    if reg_in:
        e = nearest_center(reg_in)
        _aim_at(gp, pbox, e)
        return st, inr_of(e), "shoot", len(regs)

    # 2) 幽灵(无射程内敌人时才处理)
    if ghosts:
        t = nearest_center([g[0] for g in ghosts])
        tid = next((g[1] for g in ghosts if g[0] == t), None)
        # lock 防抖
        if lock is not None:
            gc = lock.get("gc")
            keep = gc is not None and (
                any((gc[0] - g2[0][0]) ** 2 + (gc[1] - g2[0][1]) ** 2 < 80 * 80 for g2 in ghosts)
                or (time.time() - lock.get("ts", 0.0) < GHOST_DWELL_SEC))
            if keep:
                t = gc
                tid = lock.get("id")
        if lock is not None:
            lock["gc"] = t
            lock["ts"] = time.time()
            lock["id"] = tid
        _aim_at(gp, pbox, t)
        mode = "hat" if tid in HAT_GHOST_IDS else "ghost"
        return st, None, mode, len(ghosts)
    if lock is not None:
        lock.pop("gc", None)

    # 3) 普通敌人(射程外)：瞄着等
    if regs:
        e = nearest_center(regs)
        _aim_at(gp, pbox, e)
        return st, inr_of(e), "wait", len(regs)

    # 4) 大电锯(事件)
    if saw_big:
        s = nearest_center(saw_big)
        _aim_at(gp, pbox, s)
        return st, inr_of(s), "saw", len(saw_big)

    reset_controls(gp)
    return st, None, "idle", 0


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
        lock = {}
        ammo_est = None
        reload_until = 0.0
        prev_gray = None
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
                dead = boat_run >= HEART_DEAD_FRAMES and hearts <= 0   # 船消失+血量空 才算真死
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
                st, inr, mode, n_enemy = policy(res, frame, gp, pbox, prev_state, lock)
                prev_state = st

                # ---- M0b 109维状态(弹幕栅格参与决策) ----
                detects = []
                if res is not None and res[0].boxes is not None:
                    for bb in res[0].boxes:
                        x1, y1, x2, y2 = [float(v) for v in bb.xyxy[0].tolist()]
                        detects.append((int(bb.cls[0]), x1, y1, x2, y2))
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                st109 = m0b_state(frame, detects, prev_state=prev_state,
                                  prev_gray=prev_gray, hearts=hearts, include_drops=False)
                prev_gray = gray
                grid = st109["grid"]

                # ---- 内部弹药估计(视觉看不清时也能维持装填纪律) ----
                if st.get("ammo") is not None and st["ammo"] >= 1:
                    ammo_est = st["ammo"]
                if ammo_est is not None and ammo_est == 0 and time.time() >= reload_until:
                    ammo_est = 3                       # 装填完成
                eff = ammo_est if ammo_est is not None else st.get("ammo")
                eff_ok = eff is None or eff >= 1        # 未知=允许(暴风雨等场景)

                # 开火判定：shoot/saw(在射程或未知) 与 hat(打帽)；ghost/wait/idle 不扣扳机
                if mode == "ghost" or mode == "wait" or mode == "idle":
                    can_fire = False
                elif mode == "hat":
                    can_fire = eff_ok                    # 打帽阶段允许开枪
                else:                                    # shoot / saw
                    can_fire = eff_ok and inr is not False

                # ---- 风险感知(dodge)：朝向敌人的扇区弹幕威胁高 → 转向低危扇区、不射击 ----
                dodge = False
                ad = st.get("aim_deg")
                if can_fire and ad is not None and len(grid) >= 48:
                    sec = int((((ad % 360.0) + 360.0) % 360.0) // 22.5) % 16

                    def sec_threat(s):
                        return float(grid[s * 3] + grid[s * 3 + 1] + grid[s * 3 + 2])

                    t_cur = sec_threat(sec)
                    best = min(((s, sec_threat(s)) for s in range(16) if s != sec),
                               key=lambda x: x[1])
                    if t_cur > 0.18 and best[1] <= t_cur - 0.08:
                        dodge = True
                        can_fire = False
                        dang = -180.0 + best[0] * 22.5 + 11.25
                        rad = float((st.get("radius") or 700.0) * 0.6)
                        tx = pbox[0] + rad * math.cos(math.radians(dang))
                        ty = pbox[1] + rad * math.sin(math.radians(dang))
                        _aim_at(gp, pbox, (tx, ty))
                        mode = "dodge"

                if can_fire:
                    last_hit = time.time()
                    if (time.time() - last_shot) >= FIRE_PERIOD_SEC:
                        gp.right_trigger_float(1.0); gp.update(); time.sleep(RT_PULSE_SEC)
                        gp.right_trigger_float(0.0); gp.update()
                        last_shot = time.time()
                        if ammo_est is None:
                            ammo_est = 3                 # 未知弹药按满3计
                        ammo_est -= 1
                        if ammo_est <= 0:
                            ammo_est = 0
                            reload_until = time.time() + RELOAD_SEC
                if args.verbose and (time.time() - last_log) > 1.0:
                    last_log = time.time()
                    aim = round(st["aim_deg"], 1) if st["aim_deg"] is not None else None
                    c = st109["counts"]
                    gmax = int(np.argmax(grid)) if len(grid) and max(grid) > 0 else -1
                    dsec = ((gmax // 3) * 22.5) - 180 + 11.25 if gmax >= 0 else None
                    print(f"  [DBG] 模式={mode} aim={aim} 半角={st['half']} inr={inr} "
                          f"开火={can_fire} 闪避={dodge}")
                    print(f"        M0b109: 血={st109['hearts']} 弹={st109['ammo']} "
                          f"敌={st109['n_enemy']} 敌情={c} 近敌在射程={sum(r['inr'] for r in st109['nearest'])} "
                          f"弹数={st109['bullets']} 危险扇区角={dsec if dsec is not None else '-'} "
                          f"候选掉落={len(st109['drops'])}")
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
