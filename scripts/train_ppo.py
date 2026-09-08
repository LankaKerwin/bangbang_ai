# -*- coding: utf-8 -*-
"""
train_ppo.py — PPO 训练骨架 (stable-baselines3)
====================================================
用法(填完 rl_env 的 TODO 后):
    pip install stable-baselines3 gymnasium
    python scripts/train_ppo.py --timesteps 1_000_000

- 环境: scripts/rl_env.BangBangEnv
- 策略: MLP(159 观测 → 连续动作) + PPO
- 训练期可同时开着 capture_bigtext.py 攒大字样本
"""
import os
import sys
import time
import argparse

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE, "models", "release", "bb_yolov8s_v1.pt")
SAVE_DIR = os.path.join(BASE, "models", "rl")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--timesteps", type=int, default=1_000_000)
    parser.add_argument("--region", default=None, help="游戏区域 left,top,w,h(默认全屏)")
    parser.add_argument("--total_timesteps_placeholder", help="忽略(占位)")
    parser.add_argument("--ckpt_freq", type=int, default=2048,
                        help="每隔 N 步存一次中途模型到 models/rl(默认2048; 0=关闭)")
    parser.add_argument("--resume", default=None,
                        help="从已有模型 zip 续训(默认None=从零)")
    parser.add_argument("--no_fire_since", action="store_true",
                        help="env 用 159 维(无 fire_since), 配 bc_train --no_fire_since 产出的模型")
    args = parser.parse_args()

    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import DummyVecEnv
    from stable_baselines3.common.monitor import Monitor

    from rl_env import BangBangEnv

    os.makedirs(SAVE_DIR, exist_ok=True)

    env = DummyVecEnv([lambda: Monitor(BangBangEnv(
        MODEL_PATH, region=args.region, use_fire_since=not args.no_fire_since))])
    if args.resume:
        model = PPO.load(args.resume, env=env, device="cpu")
        print(f"已加载续训: {args.resume} (已有 {int(model.num_timesteps)} 步)")
    else:
        # device="cpu": 小 MLP 在 CPU 上训练即可；GPU 留给 游戏渲染+YOLO(8GB 显存三家分会 OOM/卡顿)
        model = PPO("MlpPolicy", env, verbose=1, n_steps=2048, batch_size=256,
                    learning_rate=3e-4, gamma=0.99, gae_lambda=0.95,
                    clip_range=0.2, ent_coef=0.01, n_epochs=10, device="cpu")

    from stable_baselines3.common.callbacks import CheckpointCallback

    callbacks = None
    if args.ckpt_freq > 0:
        callbacks = CheckpointCallback(save_freq=args.ckpt_freq, save_path=SAVE_DIR,
                                       name_prefix="bb_ppo_ckpt")
        print(f"中途存档已开: 每 {args.ckpt_freq} 步存一次到 {SAVE_DIR}")

    print(f"开始训练 {args.timesteps} 步... (模型存 {SAVE_DIR})")
    t0 = time.time()
    model.learn(total_timesteps=args.timesteps, callback=callbacks)
    dt = time.time() - t0
    steps = int(model.num_timesteps)
    print(f"训练完成, 实际 {steps} 步, 用时 {dt:.0f}s, 平均 {steps / max(dt, 0.1):.2f} 步/秒")
    model.save(os.path.join(SAVE_DIR, "bangbang_ppo"))
    env.close()
    print("训练完成，模型: models/rl/bangbang_ppo.zip")


if __name__ == "__main__":
    main()
