# -*- coding: utf-8 -*-
"""
exp_fire_ctx.py — 实验: 给 BC 加时序特征能否学会开火时机?
===========================================================
对照: 同样小 MLP + 同样数据量/epoch, 输入分别用
  A. obs 159                      (当前 BC)
  B. obs 159 + fire_since 160     (距上次开火帧数/15, 由 demo 动作序列离线推出)
比较: fire 开火帧灵敏度 / 误触发率 / 期望意图率 (30 次采样模拟)
用途: 决策 BC 是否值得升级为带射击上下文的观测(env 侧 _last_shot 可得, 无需重录 demo)
"""
import os
import sys
import glob
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEMO_DIR = os.path.join(BASE, "data", "demo")
FIRE_ON = 0.1
CAP = 15                 # fire_since 上限帧(≈装填2.5s + 余量 @6.5Hz)

torch.manual_seed(0)
np.random.seed(0)


def fire_since_seq(actions, cap=CAP):
    """距上次 fire>0.1 的帧数, 归一化 [0,1]。按文件独立(近似每文件单次会话)。"""
    fire = actions[:, 2] > FIRE_ON
    d = np.ones(len(actions), dtype=np.float32) * cap
    last = -10 ** 9
    for t in range(len(actions)):
        if fire[t]:
            d[t] = 0.0
            last = t
        else:
            d[t] = min(float(t - last), cap) / cap
    return d


def load(with_ctx):
    files = sorted(glob.glob(os.path.join(DEMO_DIR, "*.npz")))
    ss, aa = [], []
    for p in files:
        z = np.load(p, allow_pickle=True)
        ss.append(z["states"].astype(np.float32))
        aa.append(z["actions"].astype(np.float32))
    st = np.concatenate(ss)
    ac = np.concatenate(aa)
    ctx = []
    if with_ctx:
        off = 0
        for p in files:
            z = np.load(p, allow_pickle=True)
            n = len(z["states"])
            ctx.append(fire_since_seq(z["actions"].astype(np.float32)))
            off += n
        ctx = np.concatenate(ctx)
        st = np.concatenate([st, ctx[:, None]], axis=1)
    return st, ac


class Net(nn.Module):
    def __init__(self, dim_in):
        super().__init__()
        self.f = nn.Sequential(nn.Linear(dim_in, 128), nn.Tanh(),
                               nn.Linear(128, 128), nn.Tanh())
        self.aim = nn.Linear(128, 2)
        self.fire = nn.Linear(128, 1)

    def forward(self, x):
        h = self.f(x)
        return torch.cat([self.aim(h), self.fire(h)], dim=1)


def run(with_ctx, epochs=14, lr=1e-3, bs=256):
    st, ac = load(with_ctx)
    n = len(st)
    idx = np.random.RandomState(0).permutation(n)
    n_tr = int(0.9 * n)
    tr_i, ev_i = idx[:n_tr], idx[n_tr:]
    mean = st[tr_i].mean(0)
    std = np.maximum(st[tr_i].std(0), 1e-3)
    st = (st - mean) / std

    tgt = np.zeros_like(ac)
    tgt[:, :2] = np.clip(ac[:, :2], -1, 1)
    tgt[:, 2] = np.where(ac[:, 2] > FIRE_ON, 1.0, -1.0)
    pos = float((tgt[:, 2] > 0).mean())
    w = np.where(tgt[:, 2] > 0, 1.0 / max(pos, 1e-3), 1.0)

    Xtr = torch.from_numpy(st[tr_i]); Ytr = torch.from_numpy(tgt[tr_i]); Wtr = torch.from_numpy(w[tr_i])
    Xev = torch.from_numpy(st[ev_i]); Yev = torch.from_numpy(tgt[ev_i])
    dl = DataLoader(TensorDataset(Xtr, Ytr, Wtr), batch_size=bs, shuffle=True)

    net = Net(st.shape[1])
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    fire_t = (Yev[:, 2].numpy() > 0)
    best = None
    for ep in range(epochs):
        for x, y, ww in dl:
            opt.zero_grad()
            out = net(x)
            sw = torch.ones_like(out); sw[:, 2] = ww
            loss = torch.mean(sw * (out - y) ** 2)
            loss.backward(); opt.step()
        with torch.no_grad():
            vl = torch.mean((net(Xev) - Yev) ** 2).item()
        if best is None or vl < best[0]:
            best = (vl, {k: v.detach().clone() for k, v in net.state_dict().items()})
    net.load_state_dict(best[1])

    # 采样模拟 30 次
    with torch.no_grad():
        mu = net(Xev)
        intent = 0.0
        for _ in range(30):
            samp = mu + torch.randn_like(mu) * 0.368
            intent += (samp[:, 2] > 0).float()
        intent = intent / 30
        mu_np = mu.numpy()
    aim_err = float(np.abs(mu_np[:, :2] - Yev.numpy()[:, :2]).mean())
    sens = 100.0 * intent[fire_t].mean().item()
    false_r = 100.0 * intent[~fire_t].mean().item()
    int_rate = 100.0 * intent.mean().item()
    print("输入%d维 | 灵敏度(开火帧触发): %.0f%% | 误触发: %.1f%% | 意图率: %.1f%% | aim_err %.3f"
          % (st.shape[1], sens, false_r, int_rate, aim_err))
    return sens, false_r, int_rate


if __name__ == "__main__":
    print("== 对照: 开火时序特征有无 ==")
    print("A. 仅 obs(159):")
    a = run(False)
    print("B. obs + fire_since(160):")
    b = run(True)
    print("\n结论: 灵敏度 %+.0fpp | 误触发 %+.1fpp | 意图率 %+.1fpp"
          % (b[0] - a[0], b[1] - a[1], b[2] - a[2]))
