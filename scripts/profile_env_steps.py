# -*- coding: utf-8 -*-
"""profile_env_steps.py — 跑 N 步环境, 供 cProfile 定位每步耗时大头。
用法:
    python -m cProfile -s cumtime scripts\\profile_env_steps.py
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from rl_env import BangBangEnv

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def main():
    env = BangBangEnv(os.path.join(BASE, "models", "release", "bb_yolov8s_v1.pt"))
    obs, _ = env.reset()
    for i in range(40):
        a = np.array([0.0, 0.0, 0.0], dtype=np.float32)   # 不动不射, 测感知纯开销
        obs, r, done, trunc, info = env.step(a)
        if done or trunc:
            obs, _ = env.reset()
    env.close()
    n = max(env._n_acc, 1)
    print("PROFILE_DONE 40 steps")
    print("每步平均耗时(s):")
    for k in ("act", "sense", "obs", "shop", "sleep"):
        print("  %-6s %.4f" % (k, env._acc[k] / n))
    print("  合计   %.4f  (≈%.2f 步/秒)" % (sum(env._acc.values()) / n,
                                          n / max(sum(env._acc.values()), 1e-9)))


if __name__ == "__main__":
    main()
