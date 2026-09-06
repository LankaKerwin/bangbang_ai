# -*- coding: utf-8 -*-
"""
rl_env.py — PPO 训练环境 (Bang Bang Barrage)
====================================================
把 真实游戏 包成 gym 环境：
  观测: perception.m0b_vector (159 维, 由 m0b_state 汇总 YOLO+弹幕栅格+障碍+伤害记忆)
  动作: 连续 [aim_x, aim_y] (右摇杆) + [fire] (> FIRE_THRESH = 扣扳机意图, 半自动)
  奖励: v0 = 存活 +0.001/步, 每掉1心 -1.0, 死亡额外 -5.0
        (待与实时HP/结算OCR校准后调优; 数值集中在本文件顶部便于调整)
  回合: 判死(船消失+血量空, 照 train_driver) → done → reset() 自动重开新一局

照搬 train_driver.py 已验证的细节:
  - 内部弹药估计 + 装填计时(RELOAD_SEC) + 半自动 RT 脉冲(FIRE_PERIOD/RT_PULSE)
  - 伤害记忆: 血量下降记录时间与朝向(hurt_recent/hurt_from), 喂给 m0b_state
  - 商店: 检测到顶部 SHOP → 自动 B+A 退出(不购买), 防卡局
  - 死亡自重启: 长按A跳过加载 + 逐秒验证直到血量回来

用法:
    pip install stable-baselines3 gymnasium
    python scripts/train_ppo.py --timesteps 10_000      # v0 冒烟

注意:
  - 训练前请把游戏开进一局(或停在死亡/结算画面), 本环境会自动重开
  - include_drops=False: 掉落物检测仍是噪声候选, v0 不参与观测/奖励
"""
import os
import time
import numpy as np

try:
    import gymnasium as gym
    from gymnasium import spaces
except Exception:                     # 老版本回退
    import gym as gym
    from gym import spaces

import cv2
import mss
import vgamepad as vg

from perception import m0b_state, m0b_vector
from auto_restart import _count_solid_hearts   # 复用血量计数(实心心, 颜色无关)
from score_ocr import read_score              # 右上角分数OCR(击杀奖励信号)
from bigtext_detect import detect_reward_time # 中央大字检测(Boss击杀=奖励时间公告)

OBS_DIM = 159

# ================= 可调参数(照 train_driver) =================
CONF = 0.25                 # YOLO 置信度
IMG_SIZE = 640              # YOLO 推理尺寸
PLAYER_CLS = 0              # 玩家船类别
HEART_DEAD_FRAMES = 6       # 船连续消失帧数(且血量==0) → 判死
RELOAD_SEC = 2.5            # 打空后装填耗时估计
FIRE_PERIOD_SEC = 0.4       # 半自动: 每次扣扳机(RT)最小间隔
RT_PULSE_SEC = 0.08         # 每次扣扳机按住时长(游戏=半自动, 一按一发)
FIRE_THRESH = 0.0           # fire 动作 > 0 = 扣扳机意图(负=松开)
HURT_DECAY_SEC = 1.5        # 受伤记忆衰减时长
# ---- 商店自动退出(不购买) ----
SHOP_TPL = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                        "assets", "templates", "shop.png")
SHOP_PERSIST = 3            # SHOP 匹配连续帧数 → 判定在商店
SHOP_SCORE = 0.60
SHOP_SKIP_INTERVAL = 2.5    # 商店退出动作最小间隔(s)
# ---- 死亡自重启 ----
HOLD_SKIP_SEC = 2.5
A_PRESS_SEC = 0.15
RESTART_VERIFY_TRIES = 12   # 重启后最多逐秒验证次数
MAX_RESTART_ATTEMPTS = 2
AFTER_RESTART_WAIT = 1.0
# ================= v0 奖励(待校准) =================
REWARD_SURVIVE = 0.001      # 每步存活
REWARD_HP_LOST = -1.0       # 每损失 1 颗心
REWARD_DEATH = -5.0         # 回合结束(死亡)额外惩罚
REWARD_PER_POINT = 0.005    # 每涨1分奖励; 挡位单杀分 白60/蓝180/黄360/红600
                            # → 单杀奖励 ≈ 0.3/0.9/1.8/3.0 (鼓励冲高挡, 分数越高奖励越多)
SCORE_SAMPLE_EVERY = 5      # 每N步采样一次分数(分数单调不减, Δ累计总奖励不变, 省OCR开销)
REWARD_BOSS_KILL = 50.0     # Boss击杀(奖励时间公告出现)一次性奖励 = 结算+10000分等值(×0.005)
# ============================================================


def imread_u(path):
    """cv2.imread 中文路径不可靠 → open+imdecode。"""
    with open(path, "rb") as f:
        return cv2.imdecode(np.frombuffer(f.read(), np.uint8), cv2.IMREAD_COLOR)


def reset_controls(gp):
    """左右摇杆+左右扳机全部归中/松开。"""
    gp.left_joystick_float(0.0, 0.0)
    gp.right_joystick_float(0.0, 0.0)
    gp.left_trigger_float(0.0)
    gp.right_trigger_float(0.0)
    gp.update()


def skip_shop(gp):
    """默认不买商店: 连按 B(返回) 和 A(确认) 关掉商店退回游戏。"""
    for _ in range(2):
        gp.press_button(vg.XUSB_BUTTON.XUSB_GAMEPAD_B); gp.update(); time.sleep(0.20)
        gp.release_button(vg.XUSB_BUTTON.XUSB_GAMEPAD_B); gp.update(); time.sleep(0.25)
        gp.press_button(vg.XUSB_BUTTON.XUSB_GAMEPAD_A); gp.update(); time.sleep(0.12)
        gp.release_button(vg.XUSB_BUTTON.XUSB_GAMEPAD_A); gp.update(); time.sleep(0.25)
    reset_controls(gp)


def detect_shop(frame, tpl):
    """顶部中央匹配 SHOP 招牌, 返回相关系数(外层做持续帧判定)。"""
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


class BangBangEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, model_path, region=None, fps=10, include_drops=False):
        super().__init__()
        from ultralytics import YOLO
        self.model = YOLO(model_path)
        self.gp = vg.VX360Gamepad()
        self.region = region
        self.interval = 1.0 / fps
        self.include_drops = include_drops

        # 连续动作: aim_x, aim_y ∈[-1,1]; fire∈[-1,1](>0=扣扳机意图)
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(3,), dtype=np.float32)
        self.observation_space = spaces.Box(low=-5.0, high=5.0,
                                            shape=(OBS_DIM,), dtype=np.float32)

        self._sct = mss.mss()
        self._monitor = None
        self._z_hp = None                  # 左上血量区(按帧尺寸比例, 懒初始化)
        self._shop_tpl = imread_u(SHOP_TPL) if os.path.exists(SHOP_TPL) else None

        # 帧间记忆(每回合 reset 会清一遍)
        self._prev_state = None
        self._prev_gray = None
        self._prev_hp = None
        self._ammo_est = None              # 内部弹药估计(视觉看不清时的纪律)
        self._ammo_vis = None              # 最近一帧的视觉弹药读数
        self._reload_until = 0.0
        self._last_shot = 0.0
        self._hurt_since = -1e9
        self._hurt_from = None             # 受伤时朝向(0-1 角度归一)
        self._aim_norm = None              # 最近一帧的瞄准朝向(0-1)
        self._boat_run = 0
        self._shop_run = 0
        self._last_shop_act = 0.0
        self._prev_dead = False

    # ---------- 采集/工具 ----------
    def _grab(self):
        if self._monitor is None:
            if self.region:
                l, t, w, h = [int(x) for x in self.region.split(",")]
                self._monitor = {"left": l, "top": t, "width": w, "height": h}
            else:
                self._monitor = self._sct.monitors[1]
        f = np.array(self._sct.grab(self._monitor))
        return cv2.cvtColor(f, cv2.COLOR_BGRA2BGR)

    def _zones(self, frame):
        if self._z_hp is None:
            h, w = frame.shape[:2]
            self._z_hp = (int(w * 0.015), int(h * 0.015), int(w * 0.20), int(h * 0.085))
        return self._z_hp

    def _sense(self, frame):
        """YOLO 推理 + 血量计数 → (detects, pbox, boat, hearts)。
        detects: [(cls, x1,y1,x2,y2), ...] 含玩家; pbox 无玩家时回退画面中心。"""
        h, w = frame.shape[:2]
        hearts = _count_solid_hearts(frame, self._zones(frame))
        res = self.model.predict(frame, imgsz=IMG_SIZE, conf=CONF, verbose=False)
        detects, boat, pbox = [], False, (w / 2, h / 2)
        bx = res[0].boxes
        if bx is not None and len(bx) > 0:
            for bb in bx:
                cid = int(bb.cls[0])
                x1, y1, x2, y2 = [float(v) for v in bb.xyxy[0].tolist()]
                detects.append((cid, x1, y1, x2, y2))
                if cid == PLAYER_CLS:
                    boat = True
                    pbox = ((x1 + x2) / 2, (y1 + y2) / 2)
        return detects, pbox, boat, hearts

    def _observe(self, frame, detects, hearts, hurt_recent, hurt_from):
        """YOLO detects → m0b_state → 159 维向量(照 train_driver 的组装方式)。"""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        st = m0b_state(frame, detects, prev_state=self._prev_state,
                       prev_gray=self._prev_gray, hearts=hearts,
                       include_drops=self.include_drops,
                       hurt_recent=hurt_recent, hurt_from=hurt_from)
        self._prev_state = st
        self._prev_gray = gray
        self._aim_norm = st["aim_norm"]                  # 供下一帧受伤记忆
        if st["ammo"] is not None and st["ammo"] >= 1:   # 视觉正读数 → 校准内部估计
            self._ammo_est = st["ammo"]
        self._ammo_vis = st["ammo"]
        return m0b_vector(st), st

    def _is_dead(self, boat, hearts):
        """判死(颜色无关): 玩家船连续消失 ≥HEART_DEAD_FRAMES 且 血量==0。"""
        self._boat_run = self._boat_run + 1 if not boat else 0
        return self._boat_run >= HEART_DEAD_FRAMES and hearts <= 0

    def _fire(self):
        """半自动扣扳机: 一按一发(RT 脉冲), 受内部弹药/装填/频率纪律约束。"""
        if self._ammo_est is not None and self._ammo_est == 0:
            return False                       # 装填中, 干打会打断装填(不断连击但浪费)
        if time.time() - self._last_shot < FIRE_PERIOD_SEC:
            return False
        self.gp.right_trigger_float(1.0); self.gp.update(); time.sleep(RT_PULSE_SEC)
        self.gp.right_trigger_float(0.0); self.gp.update()
        self._last_shot = time.time()
        if self._ammo_est is None:
            self._ammo_est = 3                 # 未知弹药按满 3 计
        self._ammo_est -= 1
        if self._ammo_est <= 0:
            self._ammo_est = 0
            self._reload_until = time.time() + RELOAD_SEC
        return True

    def _in_shop(self, frame):
        if self._shop_tpl is None:
            return False
        score = detect_shop(frame, self._shop_tpl)
        self._shop_run = self._shop_run + 1 if score > SHOP_SCORE else 0
        return self._shop_run >= SHOP_PERSIST

    def _leave_shop(self):
        """检测到商店且非死亡 → 自动 B+A 退出(不购买), 防卡局。"""
        if time.time() - self._last_shop_act < SHOP_SKIP_INTERVAL:
            reset_controls(self.gp)
            return
        print("[环境·商店] 自动退出(连按B+A), 不购买")
        skip_shop(self.gp)
        self._last_shop_act = time.time()

    def _restart_round(self):
        """长按A跳过结算加载 → 逐秒验证并补按A, 直到新一局血量回来(照 train_driver)。"""
        reset_controls(self.gp)
        time.sleep(0.3)
        print("  [环境·重开] 长按A跳过加载(%ss)..." % HOLD_SKIP_SEC)
        self.gp.press_button(vg.XUSB_BUTTON.XUSB_GAMEPAD_A); self.gp.update()
        time.sleep(HOLD_SKIP_SEC)
        self.gp.release_button(vg.XUSB_BUTTON.XUSB_GAMEPAD_A); self.gp.update()
        for i in range(RESTART_VERIFY_TRIES):
            time.sleep(1.0)
            fr = self._grab()
            hs = _count_solid_hearts(fr, self._zones(fr))
            print(f"    t{i}: hearts={hs} 仍死亡={hs <= 0}")
            if hs > 0:
                print("  ✓ 新一局已开始")
                return True
            self.gp.press_button(vg.XUSB_BUTTON.XUSB_GAMEPAD_A); self.gp.update()
            time.sleep(A_PRESS_SEC)
            self.gp.release_button(vg.XUSB_BUTTON.XUSB_GAMEPAD_A); self.gp.update()
        return False

    # ---------- gym 接口 ----------
    def reset(self, *, seed=None, options=None):
        """开始/继续一局: 血量已回来 → 直接返回观测(不误按 A);
        死亡/结算画面 → 自动重开(长按A+补按A直到血量回来)。"""
        try:
            super().reset(seed=seed)
        except TypeError:            # 旧版 gym 的 reset() 不接受 seed
            super().reset()
        reset_controls(self.gp)
        # 清帧间记忆(新一局)
        self._prev_state = None
        self._prev_gray = None
        self._prev_hp = None
        self._ammo_est = None
        self._ammo_vis = None
        self._reload_until = 0.0
        self._last_shot = 0.0
        self._hurt_since = -1e9
        self._hurt_from = None
        self._aim_norm = None
        self._boat_run = 0
        self._shop_run = 0
        self._prev_dead = False
        self._prev_score = None      # 分数OCR基线(击杀奖励)
        self._reward_time_on = False # 是否在奖励时间中(防重复给Boss奖励)
        self._steps_alive = 0        # 本局已走步数(回合结束诊断)
        self._round_t0 = time.time()

        fr = self._grab()
        if _count_solid_hearts(fr, self._zones(fr)) <= 0:
            for attempt in range(1, MAX_RESTART_ATTEMPTS + 1):
                print(f"[环境·reset] 血量为空(死亡/结算), 尝试重开 #{attempt}")
                if self._restart_round():
                    break
            time.sleep(AFTER_RESTART_WAIT)

        # 用当前帧生成首观测
        frame = self._grab()
        detects, pbox, boat, hearts = self._sense(frame)
        obs, st = self._observe(frame, detects, hearts, hurt_recent=0.0, hurt_from=None)
        self._prev_hp = hearts
        info = {"time": time.time(), "hp": hearts, "ammo": self._ammo_est,
                "n_enemy": st["n_enemy"], "dead": False}
        return obs, info

    def step(self, action):
        t0 = time.time()
        # 1) 执行动作: 右摇杆瞄准 + fire>0 → 半自动扣扳机(装填纪律照 train_driver)
        if self._prev_dead:
            reset_controls(self.gp)
        else:
            if self._ammo_est is not None and self._ammo_est == 0 \
                    and time.time() >= self._reload_until:
                self._ammo_est = 3           # 装填完成
            # 瞄准摇杆: 左摇杆(真人操作习惯; 游戏双摇杆等效, 录制BC数据也用左摇杆, 字面一致)
            self.gp.left_joystick_float(
                x_value_float=float(np.clip(action[0], -1, 1)),
                y_value_float=float(np.clip(action[1], -1, 1)))
            self.gp.update()
            if float(action[2]) > FIRE_THRESH:
                self._fire()

        # 2) 抓帧 → 感知(血量/YOLO) → 判死(船消失+血量空)
        frame = self._grab()
        detects, pbox, boat, hearts = self._sense(frame)
        dead = self._is_dead(boat, hearts)

        # 3) 受伤记忆 + 奖励
        dhp = 0
        if self._prev_hp is not None:
            dhp = max(0, self._prev_hp - hearts)
            if 0 < dhp <= 3 and self._prev_hp >= 2:    # 防读数抖动: 仅记合理单次掉血
                self._hurt_since = time.time()
                self._hurt_from = self._aim_norm        # 受伤时(上一帧)朝向
        hurt_recent = max(0.0, 1.0 - (time.time() - self._hurt_since) / HURT_DECAY_SEC)
        obs, st = self._observe(frame, detects, hearts, hurt_recent,
                                self._hurt_from if self._hurt_from is not None else None)
        self._prev_hp = hearts

        self._steps_alive += 1          # 回合存活计步
        reward = REWARD_SURVIVE
        if dhp > 0:
            reward += REWARD_HP_LOST * dhp
        if dead:
            reward += REWARD_DEATH
            print("[回合结束] 存活%d步 %.1fs (hp=%d)" % (self._steps_alive,
                  time.time() - self._round_t0, hearts))

        # 分数OCR → 击杀奖励: 涨分即正反馈; 采样周期SCORE_SAMPLE_EVERY减开销
        # (死亡帧/读不出跳过; 跨采样窗口的涨分合并给奖励, 总奖励一致)
        if not dead and self._steps_alive % SCORE_SAMPLE_EVERY == 0:
            score = read_score(frame)
            if score is not None:
                if self._prev_score is None:
                    self._prev_score = score          # 首帧基线
                else:
                    ds = score - self._prev_score
                    if ds > 0:
                        reward += REWARD_PER_POINT * ds
                        self._prev_score = score      # 只信涨的, 防OCR抖动回退
            # Boss击杀: 中央"奖励时间"大字上升沿 → 一次性大奖励(active锁防重复)
            rt = detect_reward_time(frame)
            if rt and not self._reward_time_on:
                reward += REWARD_BOSS_KILL
                print("[Boss击杀] 奖励时间公告! +%.1f" % REWARD_BOSS_KILL)
            self._reward_time_on = rt

        # 4) 商店: 非死亡时自动退出(不购买); 死亡则摇杆清零
        in_shop = False
        if not dead and self._in_shop(frame):
            in_shop = True
            self._leave_shop()
        if dead:
            reset_controls(self.gp)

        self._prev_dead = dead

        # 5) 步频控制(与帧率对齐)
        dt = time.time() - t0
        if dt < self.interval:
            time.sleep(self.interval - dt)

        info = {"time": time.time(), "hp": hearts, "ammo": self._ammo_est,
                "dead": dead, "in_shop": in_shop, "dhp": dhp,
                "n_enemy": st["n_enemy"], "hurt_recent": hurt_recent}
        return obs, float(reward), dead, False, info

    def close(self):
        reset_controls(self.gp)
        try:
            self._sct.close()
        except Exception:
            pass


if __name__ == "__main__":
    # 冒烟自检: 不训练, 只验证 env 能 reset/step(需要游戏在前台运行)
    import sys
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    env = BangBangEnv(os.path.join(BASE, "models", "release", "bb_yolov8s_v1.pt"))
    obs, info = env.reset()
    print("reset obs:", obs.shape, "info:", info)
    for i in range(30):
        a = np.array([0.0, 0.0, 1.0], dtype=np.float32)
        obs, r, done, trunc, info = env.step(a)
        print(f"step{i}: r={r:+.3f} done={done} hp={info['hp']} ammo={info['ammo']}")
        if done:
            break
    env.close()
    print("smoke OK")
