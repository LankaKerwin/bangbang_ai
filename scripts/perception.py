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
import cv2

from ammo_detect import detect_ammo, half_angle_for_radius
from bullet_grid import danger_grid, obstacle_grid, N_SECTORS, RINGS_PX
from drop_detect import detect_drops
import bb_classes as B

_ID = B.TOKEN_TO_ID
_GHOST_IDS = {i for t, i in B.TOKEN_TO_ID.items() if t.startswith("ghost_")}
_SAW_IDS = {i for t, i in B.TOKEN_TO_ID.items() if t.startswith("saw_")}
_HAT_ID = _ID["ghost_hat"]

# ============ M0b 汇总状态 (收尾) ============
K_ENEMY = 6     # 记录最近的敌人数
K_DROP = 4      # 记录最近的掉落候选数
GRID_N = N_SECTORS * len(RINGS_PX)


def _grp(cid, big_saw):
    if cid in _GHOST_IDS:
        return 2 if cid == _HAT_ID else 1     # 2=帽子幽灵 1=普通幽灵
    if cid in _SAW_IDS:
        return 4 if big_saw else 3            # 4=大电锯(事件) 3=小电锯(可打)
    return 0                                   # 0=普通敌


def m0b_state(bgr, detections, prev_state=None, prev_gray=None,
              hearts=None, include_drops=True, hurt_recent=0.0, hurt_from=None):
    """汇总 M0b 全部感知 → 结构化状态 dict（M0b 收尾接口 + 受伤记忆/障碍栅格）。

    detections: [(cls_id, x1,y1,x2,y2), ...]（YOLO 原始框，含玩家）
    hurt_recent: 最近受伤程度(0-1衰减); hurt_from: 受伤时朝向(0-1角度归一, 可None)
    返回 dict：self/enemies/grid/obstacle/drops + 扁平向量见 m0b_vector。
    """
    h, w = bgr.shape[:2]
    player_c = None
    boxes = []
    for (cid, x1, y1, x2, y2) in detections:
        c = ((x1 + x2) / 2, (y1 + y2) / 2)
        if cid == 0:
            player_c = c
        else:
            boxes.append((cid, x1, y1, x2, y2, c))
    if player_c is None:
        player_c = (w / 2, h / 2)

    # 1) 自身：弹药/瞄准/范围
    st = state_from_frame(bgr, player_c, prev_state)

    # 2) 敌情聚合
    counts = {"reg": 0, "ghost": 0, "hat": 0, "saw_small": 0, "saw_big": 0}
    rows = []          # (cid,c,big)
    farea = h * w
    enemy_boxes = []
    for (cid, x1, y1, x2, y2, c) in boxes:
        area = (x2 - x1) * (y2 - y1)
        big = cid in _SAW_IDS and area > 0.008 * farea
        g = _grp(cid, big)
        key = {0: "reg", 1: "ghost", 2: "hat", 3: "saw_small", 4: "saw_big"}[g]
        counts[key] += 1
        enemy_boxes.append((x1, y1, x2, y2))
        rows.append((cid, c, g, area))

    near = []
    for (cid, c, g, area) in rows:
        dx = (c[0] - player_c[0]) / w
        dy = (c[1] - player_c[1]) / h
        dist = ((c[0] - player_c[0]) ** 2 + (c[1] - player_c[1]) ** 2) ** 0.5
        rng = st["radius"] or 1e9
        inr = enemy_in_range(st, c, player_c)
        near.append({"dx": float(dx), "dy": float(dy), "dist_norm": float(min(dist / rng, 3.0)),
                     "ang_norm": float(((np.degrees(np.arctan2(c[1] - player_c[1], c[0] - player_c[0])) + 180) % 360) / 360),
                     "grp": int(g), "inr": 1 if inr else 0})
    near.sort(key=lambda r: r["dist_norm"])
    near = near[:K_ENEMY]

    # 3) 弹幕危险栅格(prev_gray 给定时只计移动弹) + 障碍物粗栅格
    bullets, grid = danger_grid(bgr, player_c, prev_gray=prev_gray)
    grid_list = [float(x) for x in grid]
    obstacle = [float(x) for x in obstacle_grid(bgr, player_c)]

    # 4) 掉落候选(最近 K_DROP)
    drops = []
    if include_drops:
        for d in detect_drops(bgr, enemy_boxes=enemy_boxes):
            dx = (d["c"][0] - player_c[0]) / w
            dy = (d["c"][1] - player_c[1]) / h
            drops.append((d["type"], float(dx), float(dy)))
        drops = drops[:K_DROP]

    return {"hearts": hearts,
            "hurt_recent": float(hurt_recent),
            "hurt_from": hurt_from,
            "ammo": st["ammo"], "aim_norm": ((st["aim_deg"] if st["aim_deg"] is not None else 0.0) + 180) / 360,
            "aim_deg": st["aim_deg"], "radius": st["radius"], "half": st["half"],
            "n_enemy": len(boxes), "counts": counts,
            "nearest": near, "grid": grid_list, "obstacle": obstacle,
            "bullets": len(bullets),
            "drops": drops, "player_c": player_c}


# 扁平向量布局：总长固定 = 13 + K_ENEMY*5 + GRID_N + GRID_N(obstacle) + K_DROP*5 = 159
def m0b_vector(st):
    """把 m0b_state 转成固定长度 np.float32（159 维）。
    布局: [hearts/10, n_enemy/30, hurt_recent, hurt_from]
          [ammo, aim_norm, radius/2000, half/45]
          [n_reg,n_ghost,n_hat,n_saw_small,n_saw_big 各 /20]
          [nearest(K_ENEMY)×{ang_norm,dist_norm,grp/4,inr,dx}]
          [grid GRID_N] [obstacle GRID_N]
          [drops(K_DROP)×{dx,dy,red,blue,diamond}]
    """
    c = st["counts"]
    v = [0.0 if st["hearts"] is None else st["hearts"] / 10.0,
         st["n_enemy"] / 30.0,
         st["hurt_recent"],
         st["hurt_from"] if st["hurt_from"] is not None else 0.5,
         st["ammo"] if st["ammo"] is not None else -1.0,
         st["aim_norm"],
         (st["radius"] or 0.0) / 2000.0,
         (st["half"] if st["half"] is not None else 0.0) / 45.0,
         c["reg"] / 20.0, c["ghost"] / 20.0, c["hat"] / 20.0,
         c["saw_small"] / 20.0, c["saw_big"] / 20.0]
    for r in st["nearest"]:
        v += [r["ang_norm"], r["dist_norm"], r["grp"] / 4.0, r["inr"], r["dx"]]
    v += list(st["grid"])
    v += list(st["obstacle"])
    for _ in range(K_ENEMY - len(st["nearest"])):
        v += [0.0] * 5
    for (typ, dx, dy) in st["drops"]:
        v += [dx, dy, 1.0 if typ == "red" else 0.0,
              1.0 if typ == "blue" else 0.0, 1.0 if typ == "diamond" else 0.0]
    for _ in range(K_DROP - len(st["drops"])):
        v += [0.0] * 5
    return np.array(v, dtype=np.float32)


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
          "half": None,
          "radius": info["radius"],
          "angle_range": info["angle_deg"],
          "dots": info["dots"]}
    st["half"] = half_angle_for_radius(st["radius"])
    if info["dots"] and info["angle_deg"] is not None:
        angs = [b["a"] for b in info["dots"]]
        st["aim_deg"] = _circular_mean(angs)
    elif prev is not None and prev.get("aim_deg") is not None:
        # 0发/无点：沿用上一帧瞄准方向(装填期方向通常不变)
        st["aim_deg"] = prev["aim_deg"]
        st["radius"] = prev.get("radius")
        st["half"] = prev.get("half")
    return st


def _ang_diff(a, b):
    """两角度差(度)，归一化到 [-180,180)。"""
    d = a - b
    return (d + 180.0) % 360.0 - 180.0


def enemy_in_range(st, enemy_center, player_center, margin=0.10):
    """敌人是否在当前攻击范围内：
    ① 敌我距离 ≤ radius×(1+margin)  ② |敌人方向角 − aim_deg| ≤ half
    缺半径/方向时返回 None(未知，不瞎开火)。"""
    if st.get("radius") is None or st.get("aim_deg") is None or st.get("half") is None:
        return None
    px, py = player_center
    ex, ey = enemy_center
    dist = ((ex - px) ** 2 + (ey - py) ** 2) ** 0.5
    if dist > st["radius"] * (1.0 + margin):
        return False
    e_ang = float(np.degrees(np.arctan2(ey - py, ex - px)))
    if abs(_ang_diff(e_ang, st["aim_deg"])) > st["half"] + 2.0:
        return False
    return True
