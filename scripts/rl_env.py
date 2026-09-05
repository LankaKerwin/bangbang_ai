# -*- coding: utf-8 -*-
"""
rl_env.py — PPO 训练环境骨架 (Bang Bang Barrage)
====================================================
把 真实游戏 包成 gym 环境：
  观测: perception.m0b_vector (159 维)
  动作: 连续 [aim_x, aim_y] (右摇杆) + [fire] (阈值>0.5 触发半自动扣扳机)
  奖励: 骨架版(占位) → 由真实 HP/弹药等计算，后续与 结算OCR 校准
本文件是【骨架】：标 TODO 的地方需按真实采集补全后接 stable-baselines3 PPO。

依赖: pip install stable-baselines3 gymnasium
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

from perception import m0b_state, m0b_vector, state_from_frame

OBS_DIM = 159


class BangBangEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, model_path, region=None, fps=10):
        super().__init__()
        from ultralytics import YOLO
        self.model = YOLO(model_path)
        self.gp = vg.VX360Gamepad()
        self.region = region
        self.interval = 1.0 / fps

        # 连续动作: aim_x, aim_y ∈[-1,1]; fire∈[0,1](>0.5=扣扳机)
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(3,), dtype=np.float32)
        self.observation_space = spaces.Box(low=-5.0, high=5.0,
                                            shape=(OBS_DIM,), dtype=np.float32)

        self._sct = mss.mss()
        self._monitor = None
        self._prev_gray = None
        self._prev_state = None
        self._prev_hp = None
        self._ammo_est = None
        self._reload_until = 0.0
        self._last_shot = 0.0
        self._last_hp = None

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

    def _observe(self, frame, hearts, hurt_recent, hurt_from):
        # TODO: 把 YOLO res 汇总成 detections 并得到 159 维观测(可复制 train_driver 的做法)
        raise NotImplementedError("TODO: 组装 detections → m0b_state → m0b_vector")

    def _is_dead(self, frame):
        # TODO: 复用 train_driver 的判死(船消失+血量空) 
        raise NotImplementedError("TODO: 死亡判定")

    # ---------- gym 接口 ----------
    def reset(self, *, seed=None, options=None):
        # TODO: 需要"开新一局"(把 auto_restart 序列接进来)或等人为开局后调用来对齐
        super().reset(seed=seed)
        self._prev_state = None
        self._prev_gray = None
        self._prev_hp = None
        self._ammo_est = None
        obs = np.zeros(OBS_DIM, dtype=np.float32)
        # TODO: 进战斗画面后抓一帧生成真 obs
        return obs, {}

    def step(self, action):
        # 1) 应用动作: 右摇杆瞄准 + fire 阈值
        self.gp.right_joystick_float(
            x_value_float=float(np.clip(action[0], -1, 1)),
            y_value_float=float(np.clip(action[1], -1, 1)))
        # TODO: 弹药估计与半自动扣扳机(照 train_driver 搬)
        # 2) 抓帧 → 观测
        frame = self._grab()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # TODO: hearts/受伤记忆 → _observe(...)
        obs = np.zeros(OBS_DIM, dtype=np.float32)   # 占位
        # 3) 奖励(骨架占位 → 换成真实 HP/命中/弹药变化)
        reward = 0.0
        # TODO: reward 核心 —— 用"结算面板 OCR"和实时 HP 校准后填充
        # 4) 回合结束
        done = self._is_dead(frame) if False else False   # TODO: 接真死亡判定
        truncated = False
        info = {"time": time.time()}
        self._prev_gray = gray
        return obs, float(reward), done, truncated, info

    def close(self):
        self.gp.left_joystick_float(0, 0)
        self.gp.right_joystick_float(0, 0)
        self.gp.update()
        try:
            self._sct.close()
        except Exception:
            pass
