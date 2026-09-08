# -*- coding: utf-8 -*-
"""
bc_train.py — 人类演示 行为克隆(BC) → 产出 PPO 可续训 zip
==========================================================
数据: data\\demo\\*.npz (record_demo.py 产物; 14928 样本 / 2 份)
监督: (159 obs) → 动作 [aim_x, aim_y, fire]
  - aim: 回归录制左摇杆值 (录制时已 clip [-1,1])
  - fire: 二值化为 {+1 开火, -1 不开} (RT 压力 >0.1 视为开火意图),
          对齐 env 动作空间 Box[-1,1]; PPO 里 >0 = 扣扳机
网络: 与 SB3 MlpPolicy 完全同构 (Tanh, pi/vf 各 [64,64]) — 训练 pi 分支,
      学完把权重保留在真实 PPO 实例里 save zip → train_ppo.py --resume 直接微调
     (log_std 初始设小: 采样集中 mean 附近, 避免热身期乱开枪)

用法:
    python scripts/bc_train.py [--epochs 30] [--lr 1e-3]
                               [--out models/bc/bc_ppo]
"""
import os
import sys
import glob
import time
import argparse

import numpy as np
import torch
import torch.nn.functional as F
import gymnasium as gym
from torch.utils.data import TensorDataset, DataLoader

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEMO_DIR = os.path.join(BASE, "data", "demo")
FIRE_ON = 0.1            # RT 压力超过此值 = 一次开火意图
FPS_EST = 6.5            # demo 实际采样率估计(analyze_demo 实测; 秒换算用)
FIRE_SINCE_CAP_SEC = 3.0 # fire_since 归一化上限(同 rl_env.FIRE_SINCE_CAP_SEC)
FIRE_SINCE_SAT = 0.85    # 信息窗钳制值(必须与 rl_env._observe 一致!):
                         # fire_since 只有"冷却/装填窗口内"携带信息; 超过(=早已冷却完毕)
                         # 若仍输出 1.0, 部署时 agent 一旦不开火就恒停饱和值 → 落入训练分布
                         # 稀有区(听雨间隔>3s 的帧仅~1%)→ OOD 死锁(见 exp_selfdrive)。
                         # 钳到 0.85(≈2.55s>装填2.5s)使饱和帧回到常见分布区, 模型被迫
                         # 依赖外部状态(敌人位置等)决策, 不再被"饱和→别开"伪信号锁死。


def _fire_since_sec(actions):
    """距上次开火意图的秒数, 压缩到 [0, FIRE_SINCE_SAT]。与 rl_env 部署(真实秒)语义一致。
    单位=帧差÷估计帧率(demo npz 未存时间戳, 用平均帧率近似)。"""
    fire = actions[:, 2] > FIRE_ON
    N = len(actions)
    fs = np.full(N, FIRE_SINCE_SAT, dtype=np.float32)
    last = -10 ** 9
    for t in range(N):
        if fire[t]:
            fs[t] = 0.0
            last = t
        else:
            sec = (t - last) / FPS_EST
            fs[t] = min(sec / FIRE_SINCE_CAP_SEC, FIRE_SINCE_SAT)
    return fs


def load_demos(with_fs=True, use_ext=False):
    files = sorted(glob.glob(os.path.join(DEMO_DIR, "*.npz")))
    if not files:
        raise SystemExit("无 demo 文件: %s" % DEMO_DIR)
    from perception import ext_target_feats   # 159 → 3 外部可打性特征
    ss, aa = [], []
    for p in files:
        z = np.load(p, allow_pickle=True)
        s = z["states"].astype(np.float32)     # 原始 159
        a = z["actions"].astype(np.float32)
        if with_fs:
            s = np.concatenate([s, _fire_since_sec(a)[:, None]], axis=1)
        if use_ext:
            ext = np.stack([ext_target_feats(v) for v in s[:, :159]])
            s = np.concatenate([s, ext], axis=1)
        ss.append(s)
        aa.append(a)
    st = np.concatenate(ss)
    ac = np.concatenate(aa)
    print("数据: %d 份 demo / %d 样本 / obs %s  [fs=%s ext=%s]"
          % (len(files), len(st), st.shape, with_fs, use_ext))
    return st, ac


def make_targets(ac):
    t = np.zeros_like(ac)
    t[:, :2] = np.clip(ac[:, :2], -1.0, 1.0)          # aim 原样
    t[:, 2] = np.where(ac[:, 2] > FIRE_ON, 1.0, -1.0)  # fire 二值 ±1
    pos = float((t[:, 2] > 0).mean())
    w = np.where(t[:, 2] > 0, 1.0 / max(pos, 1e-3), 1.0)  # 正样本加权, 平衡 91:9
    print("fire 开火帧占比: %.1f%% → 正样本权重 %.1f" % (100 * pos, w.max()))
    return t.astype(np.float32), w.astype(np.float32)


from stable_baselines3.common.torch_layers import BaseFeaturesExtractor


class ZNormExtractor(BaseFeaturesExtractor):
    """obs 逐维 z-score 归一化, 作为 policy 的固定输入层。
    归一化统计(buffer)随 zip 保存 → PPO resume 时自动沿用, 训练/部署一致。
    解决 obs 各维尺度悬殊(0~1 与 0~7000+)导致梯度被大尺度维主导的问题。"""

    def __init__(self, observation_space, mean, std):
        super().__init__(observation_space,
                         features_dim=int(np.prod(observation_space.shape)))
        self.register_buffer("mean", torch.as_tensor(np.asarray(mean, np.float32)))
        self.register_buffer("std", torch.as_tensor(np.asarray(std, np.float32)))

    def forward(self, obs):
        return (obs - self.mean) / self.std


class _PlaceholderEnv(gym.Env):
    """零副作用 env: 只提供 spaces(PPO 初始化 policy 结构用, 不跑游戏)。"""
    def __init__(self, dim):
        self.observation_space = gym.spaces.Box(low=-5.0, high=5.0,
                                                shape=(dim,), dtype=np.float32)
        self.action_space = gym.spaces.Box(low=-1.0, high=1.0,
                                           shape=(3,), dtype=np.float32)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        return np.zeros(self.observation_space.shape, dtype=np.float32), {}

    def step(self, action):
        return np.zeros(self.observation_space.shape, dtype=np.float32), 0.0, False, False, {}


def build_ppo(mean, std):
    """构造与 train_ppo.py 同构的 PPO(占位 env): 归一化输入 + 加宽 pi 分支。
    policy_kwargs 随 zip 保存, resume 时 SB3 自动按它重建 → 结构与训练一致。
    输入维度由 mean 长度决定(=obs 159 + fire_since 1)。"""
    from stable_baselines3 import PPO
    ppo = PPO("MlpPolicy", _PlaceholderEnv(dim=int(len(mean))), n_steps=2048,
              batch_size=256, learning_rate=3e-4, gamma=0.99, gae_lambda=0.95,
              clip_range=0.2, ent_coef=0.01, n_epochs=10,
              device="cpu", seed=0, verbose=0,
              policy_kwargs=dict(
                  features_extractor_class=ZNormExtractor,
                  features_extractor_kwargs=dict(mean=mean, std=std),
                  net_arch=dict(pi=[256, 256], vf=[64, 64])))
    return ppo


def bc_loss(policy, obs, target, wmat):
    latent = policy.extract_features(obs)
    latent_pi = policy.mlp_extractor.forward_actor(latent)
    mean = policy.action_net(latent_pi)                # (B,3)
    # wmat: (B,3) 逐维权重 — aim 列=aim 聚焦权重; fire 列=正样本放大
    return torch.mean(wmat * (mean - target) ** 2), mean


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--out", default=os.path.join(BASE, "models", "bc", "bc_ppo"))
    ap.add_argument("--log_std", type=float, default=-1.0,
                    help="BC 后 log_std(默认-1 → std 0.37, 采样贴 mean)")
    ap.add_argument("--fire_bias", type=float, default=0.5,
                    help="保存前 fire 输出减此偏移: 压低高斯右尾翻正导致的乱开枪"
                         "(默认0.5; 0=不调)")
    ap.add_argument("--no_fire_since", action="store_true",
                    help="不加 fire_since 时序特征(159维纯外部状态): fire_since 是教师动作"
                         "衍生的自指特征, 部署会 OOD 死锁(见 exp_selfdrive); 此开关用于对照")
    ap.add_argument("--ext", action="store_true",
                    help="追加 3 维外部可打性特征(has_target/aim_miss/dist), 增强 fire-aim 联动"
                         "(与 --no_fire_since 组合 = 推荐配置 162维)")
    ap.add_argument("--aim_focus", type=float, default=0.0,
                    help="aim 聚焦: 对'附近3×弧半径内有敌人'的帧, aim 的 MSE 权重 = 1+此值"
                         "(如3)。稀释'无目标乱转'帧对 aim 的噪声, 让模型优先学有目标时往哪瞄")
    args = ap.parse_args()

    st, ac = load_demos(with_fs=not args.no_fire_since, use_ext=args.ext)
    tgt, wgt = make_targets(ac)

    # 权重矩阵 (n,3): aim 两列共享 aim 权重, fire 列用正样本放大权重
    base159 = st[:, :159]
    wmat = np.ones((len(st), 3), dtype=np.float32)
    wmat[:, 2] = wgt
    if args.aim_focus > 0:
        near = np.zeros(len(st), dtype=bool)
        for j in range(6):                       # 任一前6敌 距离 ≤ 3×弧半径(=可命中)
            near |= (base159[:, 13 + 5 * j + 1] <= 3.0)
        wmat[:, :2] = (1.0 + args.aim_focus * near.astype(np.float32))[:, None]
        print("aim 聚焦: %.0f%% 帧有近敌 → aim 权重最高 %.1f" % (100 * near.mean(), 1 + args.aim_focus))

    # 按局切分易泄漏(相邻帧同局), 这里按样本随机切即可(监督学习, 序列无关)
    n = len(st)
    idx = np.random.RandomState(0).permutation(n)
    n_tr = int(0.9 * n)
    tr_i, ev_i = idx[:n_tr], idx[n_tr:]
    # 归一化统计只从训练段算(避免 eval 泄漏); std 下限防除零。
    # 注意: 训练输入是原始 obs — z-score 由 policy 内 ZNormExtractor 完成(与部署一致)。
    mean = st[tr_i].mean(axis=0).astype(np.float32)
    std = np.maximum(st[tr_i].std(axis=0), 1e-3).astype(np.float32)
    ds = TensorDataset(torch.from_numpy(st[tr_i]), torch.from_numpy(tgt[tr_i]),
                       torch.from_numpy(wmat[tr_i]))
    dl = DataLoader(ds, batch_size=args.batch, shuffle=True)
    ev_obs = torch.from_numpy(st[ev_i])
    ev_tgt = torch.from_numpy(tgt[ev_i])
    ev_w = torch.from_numpy(wmat[ev_i])
    print("划分: 训练 %d / 验证 %d" % (n_tr, n - n_tr))

    # 基线参考: 用 eval 目标均值当常数预测的 aim_err / fire 触发率(判断学习空间)
    ev_tgt_np = ev_tgt.numpy()
    base_aim = float(np.abs(ev_tgt_np[:, :2] - ev_tgt_np[:, :2].mean(axis=0)).mean())
    print("基线 aim_err(常数预测=目标均值): %.3f → 学习空间到此为止" % base_aim)

    ppo = build_ppo(mean, std)
    pol = ppo.policy
    # 只更新 actor 通路参数 (shared/policy + action_net)。
    # 直接收 mlp_extractor 全部参数: BC loss 不经过 vf 分支 → 其梯度为 None 不会被更新。
    opt = torch.optim.Adam(
        list(pol.mlp_extractor.parameters()) + list(pol.action_net.parameters()),
        lr=args.lr)
    best = float("inf")
    best_sd = None
    best_ep = 0
    stall = 0
    for ep in range(args.epochs):
        t0 = time.time()
        tot, nb = 0.0, 0
        for ob, tg, w_ in dl:
            opt.zero_grad()
            loss, _ = bc_loss(pol, ob, tg, w_)
            loss.backward()
            opt.step()
            tot += float(loss.detach()) * len(ob)
            nb += len(ob)
        tr_loss = tot / nb
        # 验证
        with torch.no_grad():
            ev_loss, ev_mean = bc_loss(pol, ev_obs, ev_tgt, ev_w)
            m = ev_mean.numpy()
            aim_err = float(np.abs(m[:, :2] - ev_tgt_np[:, :2]).mean())
            fire_pred = m[:, 2] > 0
            fire_t = ev_tgt_np[:, 2] > 0
            tp = float((fire_pred & fire_t).sum())
            fp = float((fire_pred & ~fire_t).sum())
            prec = tp / max(tp + fp, 1)
            rec = tp / max(float(fire_t.sum()), 1)
        print("epoch %2d | train %.4f | val %.4f | aim_err %.3f | fire P %.2f R %.2f | %.1fs"
              % (ep + 1, tr_loss, float(ev_loss), aim_err, prec, rec,
                 time.time() - t0))
        if float(ev_loss) < best - 1e-4:
            best = float(ev_loss)
            best_ep = ep + 1
            best_sd = {k: v.detach().clone() for k, v in pol.state_dict().items()}
            stall = 0
        else:
            stall += 1
            if stall >= 8:
                print("验证 loss %d epoch 未改善, early stop @ epoch %d" % (stall, ep + 1))
                break

    # 恢复 best 权重(防过拟合尾段) → fire 偏置 + log_std 收紧 → 存 zip
    if best_sd is not None:
        pol.load_state_dict(best_sd)
    if args.fire_bias:
        with torch.no_grad():
            pol.action_net.bias[2] -= args.fire_bias
        print("fire 偏置已调: bias[2] -= %.2f (降误开枪)" % args.fire_bias)
    out = args.out
    if not out.endswith(".zip"):
        out += ".zip"
    os.makedirs(os.path.dirname(out), exist_ok=True)
    try:
        with torch.no_grad():
            pol.log_std.fill_(args.log_std)
        print("log_std 已设 %.2f (std=%.3f)" % (args.log_std, np.exp(args.log_std)))
    except Exception as e:
        print("设 log_std 失败(忽略): %s" % e)
    ppo.save(out)
    print("BC 模型已存: %s (best epoch %d, val loss %.4f)" % (out, best_ep, best))
    print("后续: python scripts/train_ppo.py --resume %s --timesteps 20000" % out)


if __name__ == "__main__":
    main()
