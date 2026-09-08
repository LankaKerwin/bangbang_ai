# -*- coding: utf-8 -*-
"""
exp_selfdrive.py — 复现真机 rollout 的 fire 哑火: fire_since 自指陷阱?
=====================================================================
BC 训练时 fire_since 由"听雨的动作"驱动(teacher-forcing);
真机部署时 fire_since 由"agent 自己的动作"驱动(self-driving)。
若 agent 不开火 → fire_since 恒=1.0 → obs 进入训练分布边缘 → 更不开火 → 死锁。
本脚本在 demo 观测序列上对比两种模式(感知部分同一实况, 只换 fire_since 来源):
  TF: fire_since 用听雨真值          → 意图率应≈8% (离线 verify 一致)
  SD: fire_since 由模型采样动作自驱   → 若意图率暴跌 → 实锤自指陷阱
"""
import os
import sys
import numpy as np
import torch

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(BASE, "scripts"))
sys.path.insert(0, BASE)

from bc_train import _fire_since_sec, FPS_EST, FIRE_SINCE_CAP_SEC, FIRE_SINCE_SAT
from stable_baselines3 import PPO

ZIP = os.path.join(BASE, "models", "bc", "bc_ppo.zip")
DEMO = os.path.join(BASE, "data", "demo", "demo_1788697330.npz")
N_TRAJ = 10            # 轨迹数(平均掉单次采样噪声)
DT = 1.0 / FPS_EST     # 每帧真实秒(≈0.154s)
FIRE_ON = 0.1


def load():
    z = np.load(DEMO, allow_pickle=True)
    st = z["states"].astype(np.float32)
    ac = z["actions"].astype(np.float32)
    fs_tf = _fire_since_sec(ac)                # 听雨驱动的 fire_since
    return st, ac, fs_tf


def roll(model, st, fire_since_series):
    """在 demo obs 序列上前向采样一次, 统计采样 fire>0 意图率。"""
    n = len(st)
    pol = model.policy
    fr = []
    with torch.no_grad():
        for i in range(0, n, 512):
            obs = torch.from_numpy(np.concatenate([st[i:i + 512], fire_since_series[i:i + 512, None]], axis=1))
            a = pol.get_distribution(obs).sample().numpy()
            fr.append(a[:, 2] > 0)
    return np.concatenate(fr)


def selfdrive_series(model, st, seed0=0):
    """自驱 fire_since: 起始 SAT(新局, 冷却完毕), 每帧若采样 fire>0 → 0, 否则 += dt/cap(钳 SAT)。"""
    n = len(st)
    fs = np.full(n, FIRE_SINCE_SAT, dtype=np.float32)
    pol = model.policy
    torch.manual_seed(seed0)
    cur = FIRE_SINCE_SAT
    with torch.no_grad():
        i = 0
        while i < n:
            obs = torch.from_numpy(np.concatenate([st[i:i + 1], np.array([[cur]], np.float32)], axis=1))
            a = pol.get_distribution(obs).sample().numpy()[0]
            if a[2] > 0:
                cur = 0.0
            else:
                cur = min(cur + DT / FIRE_SINCE_CAP_SEC, FIRE_SINCE_SAT)
            fs[i] = cur
            i += 1
    return fs


def stat(fr, name):
    rate = 100.0 * fr.mean()
    per_sec = fr.mean() * FPS_EST   # 意图帧占比 → 每秒意图帧数(≈开枪频率上界)
    print("%-6s 意图率 %.2f%%  (≈每 %.1f 秒一次意图; 听雨: 8.3%% ≈ 每 2.0s)"
          % (name, rate, 1.0 / max(per_sec, 1e-6)))
    return rate


if __name__ == "__main__":
    m = PPO.load(ZIP, device="cpu")
    st, ac, fs_tf = load()
    print("帧数 %d | 模型 %s" % (len(st), os.path.basename(ZIP)))

    # 1) teacher-forcing: 听雨驱动的 fire_since
    fr_tf = np.stack([roll(m, st, fs_tf) for _ in range(N_TRAJ)])
    stat(fr_tf.mean(0), "TF")

    # 2) self-driving: 模型自驱 fire_since
    fr_sd = []
    for k in range(N_TRAJ):
        fs_sd = selfdrive_series(m, st, seed0=k)
        fr_sd.append(roll(m, st, fs_sd))
    stat(np.stack(fr_sd).mean(0), "SD")

    print("\nTF=用听雨节奏喂给模型; SD=模拟 agent 自己不开火→fire_since 恒 1 的处境")
    print("若 SD << TF: 自指陷阱实锤 — 部署时 agent 不开火→状态漂移→更不敢开")
