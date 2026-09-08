# -*- coding: utf-8 -*-
"""verify_bc.py — 验证 BC zip 可被 PPO.load 加载并前向(含自定义归一化层)。
用法: python scripts/verify_bc.py [--zip models/bc/bc_ppo.zip]"""
import os
import sys
import argparse
import numpy as np
import torch

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(BASE, "scripts"))
sys.path.insert(0, BASE)

from stable_baselines3 import PPO

ap = argparse.ArgumentParser()
ap.add_argument("--zip", default=os.path.join(BASE, "models", "bc", "bc_ppo.zip"))
args = ap.parse_args()

m = PPO.load(args.zip, device="cpu")
p = m.policy
print("加载OK: %s | extractor: %s" % (os.path.basename(args.zip), type(p.features_extractor).__name__))

# 用一份真实 demo obs 前向, 看动作 mean 分布合理否(近似部署行为)
d = os.path.join(BASE, "data", "demo")
files = sorted(os.path.join(d, f) for f in os.listdir(d) if f.endswith(".npz"))
z = np.load(files[-1], allow_pickle=True)
ac = z["actions"][:2000].astype(np.float32)
dim = int(m.observation_space.shape[0])
base = z["states"][:2000].astype(np.float32)
from bc_train import _fire_since_sec  # noqa: E402
from perception import ext_target_feats  # noqa: E402
if dim == 160:      # +fire_since
    st_full = z["states"].astype(np.float32)
    st = np.concatenate([st_full, _fire_since_sec(z["actions"].astype(np.float32))[:, None]],
                        axis=1)[:2000]
elif dim == 162:    # +ext 可打性 3 维
    ext = np.stack([ext_target_feats(v) for v in base])
    st = np.concatenate([base, ext], axis=1)
else:
    st = base
print("obs 维度: %d%s" % (st.shape[1],
      {160: " (+fire_since)", 162: " (+ext)"}.get(dim, " (纯159)")))
with torch.no_grad():
    obs = torch.from_numpy(st)
    dist = p.get_distribution(obs)
    mu = dist.distribution.mean.numpy()
print("\n-- 2000 帧 demo obs 的 BC 输出 --")
print("aim mean 幅度: %.3f | fire mean 分布: [min %.2f, p25 %.2f, p50 %.2f, p75 %.2f, max %.2f]"
      % (np.sqrt(mu[:, 0] ** 2 + mu[:, 1] ** 2).mean(),
         mu[:, 2].min(), *np.percentile(mu[:, 2], [25, 50, 75]), mu[:, 2].max()))
print("fire mean>0 占比: %.1f%% (听雨真值 8.3%%)" % (100.0 * (mu[:, 2] > 0).mean()))
print("aim 真值幅度: %.3f" % np.sqrt(ac[:, 0] ** 2 + ac[:, 1] ** 2).mean())

# 采样模拟: 部署时是 N(mean, std) 采样 → 统计真实触发意图率与灵敏度
torch.manual_seed(0)
n_samp = 30
fire_t = ac[:, 2] > 0.1
with torch.no_grad():
    f_on = torch.stack([(dist.sample()[:, 2] > 0).float() for _ in range(n_samp)])
    intent = f_on.mean(axis=0)             # 每帧触发概率
print("\n-- 采样模拟 (%d 次) --" % n_samp)
print("期望意图率(采样 fire>0 帧占比均值): %.1f%% (听雨 8.3%%)" % (100.0 * intent.mean()))
print("开火帧灵敏度(真开火帧中触发概率): %.0f%%" % (100.0 * intent[fire_t].mean()))
print("误触发率(不开火帧中触发概率): %.1f%%" % (100.0 * intent[~fire_t].mean()))

# 方向误差(把向量归一化后比角度): 更有物理意义
def ang_err(a, b):
    na = a / np.maximum(np.linalg.norm(a, axis=1, keepdims=True), 1e-6)
    nb = b / np.maximum(np.linalg.norm(b, axis=1, keepdims=True), 1e-6)
    c = np.clip((na * nb).sum(axis=1), -1, 1)
    return np.degrees(np.arccos(c))
err = ang_err(mu[:, :2], ac[:, :2])
print("aim 方向误差(度): mean %.1f | p50 %.1f | p90 %.1f"
      % (err.mean(), np.percentile(err, 50), np.percentile(err, 90)))
# 基线: 全零摇杆 vs 真值
err0 = ang_err(np.zeros_like(mu[:, :2]), ac[:, :2])
print("基线(摇杆不动)方向误差(度): %.1f" % err0.mean())
