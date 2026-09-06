# -*- coding: utf-8 -*-
"""
record_demo.py — 真人示范录制 (人类 BC 数据采集)
================================================
你正常用手柄打游戏, 本脚本旁观并同步录制:
    每帧 (159维观测, 你的动作 [aim_x, aim_y, fire])
- 观测: 与 rl_env 完全相同的管道 (mss抓屏 → YOLO + 血量 + m0b_state → 159维)
- 动作: 左摇杆(瞄准, 双摇杆等效, 已统一为左) + RT(开枪, 0~1压力) — 与 env 动作空间同构
- 过滤: 死亡/结算/加载/商店画面 不产生样本(那会儿你没在战斗决策)

用法:
    python scripts/record_demo.py --seconds 900
参数:
    --seconds  录制秒数(默认900; Ctrl+C 可提前结束并保存)
    --out      输出npz(默认 data\\demo\\demo_时间戳.npz)
    --fps      采样频率(默认10, 与训练同频)
    --pad      手柄槽位(默认0)
"""
import os
import sys
import time
import argparse
import ctypes

import numpy as np
import cv2
import mss

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE)

from perception import m0b_state, m0b_vector
from auto_restart import _count_solid_hearts
from rl_env import SHOP_TPL, SHOP_SCORE, detect_shop, imread_u

CONF = 0.25
IMG_SIZE = 640
PLAYER_CLS = 0
DEADZONE = 5000            # 摇杆死区(避免微漂移)
TRIGGER_MAX = 255.0

# ---------------- XInput 读取(ctypes, 无第三方依赖) ----------------
class _GAMEPAD(ctypes.Structure):
    _fields_ = [("wButtons", ctypes.c_ushort), ("bLeftTrigger", ctypes.c_ubyte),
                ("bRightTrigger", ctypes.c_ubyte), ("sThumbLX", ctypes.c_short),
                ("sThumbLY", ctypes.c_short), ("sThumbRX", ctypes.c_short),
                ("sThumbRY", ctypes.c_short)]

class _STATE(ctypes.Structure):
    _fields_ = [("dwPacketNumber", ctypes.c_ulong), ("Gamepad", _GAMEPAD)]


def _load_xinput():
    for name in ("xinput1_4", "xinput1_3", "xinput9_1_0"):
        try:
            return ctypes.windll[name]
        except Exception:
            continue
    return None


XINPUT = _load_xinput()


def read_pad(idx=0):
    """读手柄最新状态快照。未连接→None。返回 {lx,ly,rt} 归一化。"""
    if XINPUT is None:
        return None
    st = _STATE()
    if XINPUT.XInputGetState(idx, ctypes.byref(st)) != 0:
        return None
    g = st.Gamepad
    lx = g.sThumbLX / 32767.0
    ly = g.sThumbLY / 32767.0
    if abs(g.sThumbLX) < DEADZONE:
        lx = 0.0
    if abs(g.sThumbLY) < DEADZONE:
        ly = 0.0
    return {"lx": float(np.clip(lx, -1, 1)), "ly": float(np.clip(ly, -1, 1)),
            "rt": float(g.bRightTrigger / TRIGGER_MAX)}


# ---------------- 录制器 ----------------
class DemoRecorder:
    def __init__(self, model_path, pad=0):
        from ultralytics import YOLO
        self.model = YOLO(model_path)
        self.pad = pad
        self._sct = mss.mss()
        self._monitor = None
        self._z_hp = None
        self._shop_tpl = imread_u(SHOP_TPL) if os.path.exists(SHOP_TPL) else None
        self._was_dead = True          # 起始视为死亡, 等进局
        self.states = []
        self.actions = []
        self.n_rounds = 0
        self.n_dead = 0
        self.n_shop = 0
        self._clear_mem()

    def _clear_mem(self):
        """帧间记忆清空(死亡→复活时调, 仿 rl_env.reset)"""
        self._prev_state = None
        self._prev_gray = None
        self._prev_hp = None
        self._hurt_since = -1e9
        self._hurt_from = None
        self._aim_norm = None

    def _grab(self):
        if self._monitor is None:
            self._monitor = self._sct.monitors[1]
        f = np.array(self._sct.grab(self._monitor))
        return cv2.cvtColor(f, cv2.COLOR_BGRA2BGR)

    def _zones(self, frame):
        if self._z_hp is None:
            h, w = frame.shape[:2]
            self._z_hp = (int(w * 0.015), int(h * 0.015), int(w * 0.20), int(h * 0.085))
        return self._z_hp

    def _sense(self, frame):
        """YOLO → (detects, boat, hearts)"""
        hearts = _count_solid_hearts(frame, self._zones(frame))
        res = self.model.predict(frame, imgsz=IMG_SIZE, conf=CONF, verbose=False)
        detects, boat = [], False
        bx = res[0].boxes
        if bx is not None and len(bx) > 0:
            for bb in bx:
                cid = int(bb.cls[0])
                x1, y1, x2, y2 = [float(v) for v in bb.xyxy[0].tolist()]
                detects.append((cid, x1, y1, x2, y2))
                if cid == PLAYER_CLS:
                    boat = True
        return detects, boat, hearts

    def _observe(self, frame, detects, hearts):
        """组装 159 维(仿 rl_env._observe); 返回 vec。"""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        hurt_recent = max(0.0, 1.0 - (time.time() - self._hurt_since) / 1.5)
        st = m0b_state(frame, detects, prev_state=self._prev_state,
                       prev_gray=self._prev_gray, hearts=hearts,
                       include_drops=False, hurt_recent=hurt_recent,
                       hurt_from=self._hurt_from)
        self._prev_state = st
        self._prev_gray = gray
        self._aim_norm = st["aim_norm"]
        self._prev_hp = hearts
        return m0b_vector(st)

    def _in_shop(self, frame):
        if self._shop_tpl is None:
            return False
        return detect_shop(frame, self._shop_tpl) > SHOP_SCORE

    def save(self, out_path):
        np.savez(out_path, states=np.array(self.states, dtype=np.float32),
                 actions=np.array(self.actions, dtype=np.float32),
                 n_rounds=self.n_rounds)
        print("已保存: %d 样本 / %d 局 → %s" % (len(self.states), self.n_rounds, out_path))

    def run(self, seconds, fps=10.0):
        interval = 1.0 / fps
        t_end = time.time() + seconds
        print("=" * 55)
        print("真人示范录制 | %s 秒 | 采样 %sHz | 手柄槽 %d" % (seconds, fps, self.pad))
        print("开录前请把游戏开进一局; 死亡后手动重开即可; Ctrl+C 提前保存")
        print("=" * 55)
        last_t = time.time()

        while time.time() < t_end:
            t0 = time.time()
            frame = self._grab()
            detects, boat, hearts = self._sense(frame)

            if hearts <= 0:
                # 死亡/结算/加载: 不录, 标记等复活
                self.n_dead += 1
                if not self._was_dead:
                    print("  [死亡] 本局结束, 等下一局...")
                self._was_dead = True
            else:
                if self._was_dead:
                    self._was_dead = False
                    self._clear_mem()
                    self.n_rounds += 1
                    print("  [新局 %d] 开录..." % self.n_rounds)
                if self._in_shop(frame):
                    self.n_shop += 1
                else:
                    pad = read_pad(self.pad)
                    if pad is None:
                        print("  [警告] 读不到手柄! 检查连接/槽位(可 --pad 换槽)")
                        pad = {"lx": 0.0, "ly": 0.0, "rt": 0.0}
                    vec = self._observe(frame, detects, hearts)
                    self.states.append(vec)
                    self.actions.append([pad["lx"], pad["ly"], pad["rt"]])

            if time.time() - last_t > 10:
                n = len(self.states)
                if n:
                    a = np.mean(self.actions[-100:], axis=0)
                    print("  [%d样本 局%d] 最近动作均值: aim=(%.2f,%.2f) fire=%.2f"
                          % (n, self.n_rounds, a[0], a[1], a[2]))
                else:
                    print("  [%d样本 局%d] (尚无样本: 游戏中?)" % (n, self.n_rounds))
                last_t = time.time()

            dt = time.time() - t0
            if dt < interval:
                time.sleep(interval - dt)

        print("结束: %d 样本 / %d 局 / 死亡帧%d / 商店帧%d"
              % (len(self.states), self.n_rounds, self.n_dead, self.n_shop))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seconds", type=float, default=900.0)
    ap.add_argument("--fps", type=float, default=10.0)
    ap.add_argument("--pad", type=int, default=0)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    if XINPUT is None:
        print("找不到 XInput dll")
        return

    outdir = os.path.join(BASE, "data", "demo")
    os.makedirs(outdir, exist_ok=True)
    out = args.out or os.path.join(outdir, "demo_%d.npz" % int(time.time()))

    rec = DemoRecorder(os.path.join(BASE, "models", "release", "bb_yolov8s_v1.pt"),
                       pad=args.pad)
    try:
        rec.run(args.seconds, fps=args.fps)
        rec.save(out)
    except KeyboardInterrupt:
        print("\n提前结束, 保存中...")
        rec.save(out)


if __name__ == "__main__":
    main()
