# -*- coding: utf-8 -*-
"""
analyze_demo.py — 录制的 BC 演示数据体检
========================================
纯 numpy, 不依赖 YOLO/OCR, 直接分析 data\\demo\\*.npz:
- 样本量 / 局数 / 时间跨度
- aim 分布: mean/std/min/max, 死区占比(|a|<0.05), 是否覆盖全象限
- fire 分布: 触发占比(>0.1), 触发时均值, 是否有点射节奏
- 数据质量: NaN/inf, 常量帧占比(前后 obs 完全相同的比例)
"""
import os
import sys
import glob
import numpy as np

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
D = os.path.join(BASE, "data", "demo")

def analyze(path):
    z = np.load(path, allow_pickle=True)
    st = z["states"].astype(np.float32)
    ac = z["actions"].astype(np.float32)
    n_rounds = int(z["n_rounds"]) if "n_rounds" in z else -1
    print("=" * 60)
    print("文件: %s" % os.path.basename(path))
    print("样本 %d | 局数 %d | 时长≈%.0fs @~6.5Hz"
          % (len(st), n_rounds, len(st) / 6.5))
    print("states %s | actions %s" % (st.shape, ac.shape))

    # NaN/inf
    bad = (np.isnan(st).any() or np.isinf(st).any()
           or np.isnan(ac).any() or np.isinf(ac).any())
    print("NaN/inf: %s" % ("有!!" if bad else "无"))

    ax, ay, f = ac[:, 0], ac[:, 1], ac[:, 2]
    print("\n-- aim (左摇杆, 范围[-1,1]) --")
    for nm, v in (("aim_x", ax), ("aim_y", ay)):
        print("  %s: mean=%+.3f std=%.3f min=%+.3f max=%+.3f"
              % (nm, v.mean(), v.std(), v.min(), v.max()))
    dead = (np.abs(ax) < 0.05) & (np.abs(ay) < 0.05)
    print("  死区占比(几乎不动): %.1f%%" % (100.0 * dead.mean()))
    mag = np.sqrt(ax * ax + ay * ay)
    print("  摇杆幅度均值: %.3f | >0.7(大幅甩动)占比: %.1f%%"
          % (mag.mean(), 100.0 * (mag > 0.7).mean()))

    print("\n-- fire (RT 0~1) --")
    trig = f > 0.1
    print("  触发占比: %.1f%% | 非零均值: %.2f | 全段均值: %.2f | max: %.2f"
          % (100.0 * trig.mean(), f[trig].mean() if trig.any() else 0,
             f.mean(), f.max()))
    # 点射节奏: 开枪段(触发)平均连续长度
    if trig.any():
        d = np.diff(np.concatenate([[0], trig.astype(int), [0]]))
        on_lens = np.where(d == 1)[0], np.where(d == -1)[0]
        runs = on_lens[1] - on_lens[0]
        print("  开枪段数: %d | 平均每段 %.1f 帧 (≈%.2fs)" % (len(runs), runs.mean() if len(runs) else 0, runs.mean() / 6.5 if len(runs) else 0))

    print("\n-- obs 多样性 --")
    # 相邻 obs 完全相同(疑似卡帧/暂停污染)占比
    same = (st[1:] == st[:-1]).all(axis=1).mean()
    print("  相邻 obs 完全相同占比: %.2f%%" % (100.0 * same))
    # 观测值域概览(前若干维元信息是否在动)
    print("  obs 全局 mean=%.4f std=%.4f min=%.3f max=%.3f"
          % (st.mean(), st.std(), st.min(), st.max()))
    return len(st)

def main():
    files = sorted(glob.glob(os.path.join(D, "*.npz")))
    if not files:
        print("无 demo 文件: %s" % D)
        return
    total = 0
    for p in files:
        total += analyze(p)
    print("\n" + "=" * 60)
    print("合计: %d 份 / %d 样本" % (len(files), total))

if __name__ == "__main__":
    main()
