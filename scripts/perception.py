# -*- coding: utf-8 -*-
"""
perception.py v0 — M0b 感知入口(先做 瞄准方向/范围)
=====================================================
目前实现：
  state_from_frame(bgr, player_center)
    → {ammo, aim_deg(瞄准方向,度,画面x轴0°顺时针), aim_width(扇形张角),
       radius(攻击范围), angle_range, dots}
  瞄准方向 = 弹药弧中点的方向(圆均值)；半径 = 弹药弧所在距离。
  0 发时无点 → aim_deg/radius 为 None(由调用方沿用上一帧或另行检测扇形)。

依赖 ammo_detect.detect_ammo。后续会扩展：弹幕危险栅格 / 掉落物 / 档位色。
"""
import numpy as np

from ammo_detect import detect_ammo


def _circular_mean(angles_deg):
    """多角度的圆均值(度)，处理 ±180 跨越。"""
    rad = np.radians(angles_deg)
    x = float(np.mean(np.cos(rad)))
    y = float(np.mean(np.sin(rad)))
    return float(np.degrees(np.arctan2(y, x)))


def state_from_frame(bgr, player_center, prev=None):
    """prev: 上一帧 state(dict)，用于 0发 时延续方向(可选)。
    返回 dict: ammo/aim_deg/radius/angle_range/dots
    （aim_width 暂不提供：弹药点挤在弧中部，不代表扇形张角；张角留给扇形边缘检测）"""
    info = detect_ammo(bgr, player_center,
                       prev_radius=(prev.get("radius") if prev else None))
    st = {"ammo": info["ammo"],
          "aim_deg": None,
          "aim_width": None,
          "radius": info["radius"],
          "angle_range": info["angle_deg"],
          "dots": info["dots"]}
    if info["dots"] and info["angle_deg"] is not None:
        angs = [b["a"] for b in info["dots"]]
        st["aim_deg"] = _circular_mean(angs)
    elif prev is not None and prev.get("aim_deg") is not None:
        # 0发/无点：沿用上一帧瞄准方向(装填期方向通常不变)
        st["aim_deg"] = prev["aim_deg"]
        st["radius"] = prev.get("radius")
    return st
