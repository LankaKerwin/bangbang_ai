# -*- coding: utf-8 -*-
"""
bullet_grid.py — M0b 感知：敌方弹幕危险栅格 + 掉落物 (v0 骨架)
================================================================
霰弹枪无飞行子弹(瞬发) → 本模块只需认【敌方弹】。颜色区间待用真实截图标定。

API:
    bullets, grid = danger_grid(bgr, player_center)
       grid: 16扇区 × N圈 扁平列表(0~N)，值=该格威胁(弹越近权重越大)
       bullets: [(x,y),...] 敌方弹中心
    drops = detect_drops(bgr)
       drops: [{type:'red'|'blue'|'star'|'diamond', c:(x,y)}, ...]
标定/验证见 test_percept.py(可画圈输出给人眼确认)。
"""
import cv2
import numpy as np

# ===== 敌方弹 HSV 区间(骨架初值，待标定) =====
ENEMY_BULLET_HSV = [          # (Hmin,Hmax,Smin,Vmin)
    (0, 12, 100, 120),        # 红
    (12, 22, 100, 120),       # 橙
    (22, 45, 100, 120),       # 黄
    (120, 160, 100, 100),     # 紫
]
BULLET_AREA_MIN, BULLET_AREA_MAX = 8, 600
BULLET_SQUARE_MAX = 34        # 单边像素上限

# ===== 掉落物(果实/钻石)区间初值 =====
DROP_HSV = {
    "red":    [(168, 180, 100, 120), (0, 10, 100, 120)],
    "blue":   [(95, 130, 100, 100)],
    "star":   [(20, 45, 120, 150)],       # 星果(黄)
    "diamond": [(170, 180, 80, 160), (0, 8, 80, 160)],
}
DROP_AREA_MIN, DROP_AREA_MAX = 15, 900

N_SECTORS = 16
RINGS_PX = [170, 420, 800]   # 3圈半径(px, 以玩家为中心)，待标定


def _mask_for_ranges(bgr, ranges):
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    H, S, V = hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]
    mask = np.zeros(H.shape, dtype=np.uint8)
    for (h0, h1, s0, v0) in ranges:
        if h0 <= h1:
            m = (H >= h0) & (H <= h1)
        else:                       # 跨0°的红色带
            m = (H >= h0) | (H <= h1)
        m &= (S >= s0) & (V >= v0)
        mask |= m.astype(np.uint8)
    k = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    return cv2.morphologyEx(mask * 255, cv2.MORPH_OPEN, k)


def _small_blobs(mask, amin, amax, smax):
    n, lab, stats, cents = cv2.connectedComponentsWithStats(mask, 8)
    out = []
    for i in range(1, n):
        x, y, w, h, area = stats[i]
        if area < amin or area > amax:
            continue
        if max(w, h) > smax:
            continue
        out.append((float(cents[i][0]), float(cents[i][1]), float(area)))
    return out


def detect_enemy_bullets(bgr, prev_gray=None, move_diff=8.0):
    """敌方弹：小色块。prev_gray 给定时加"移动判据"——只算在动的(滤静止碎片/手/果)。"""
    mask = _mask_for_ranges(bgr, ENEMY_BULLET_HSV)
    blobs = _small_blobs(mask, BULLET_AREA_MIN, BULLET_AREA_MAX, BULLET_SQUARE_MAX)
    if prev_gray is None:
        return blobs
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    diff = cv2.absdiff(gray, prev_gray)
    out = []
    for (cx, cy, area) in blobs:
        x0, y0 = max(0, int(cx) - 4), max(0, int(cy) - 4)
        patch = diff[y0:y0 + 9, x0:x0 + 9]
        if patch.size and float(np.mean(patch)) > move_diff:
            out.append((cx, cy, area))
    return out


def danger_grid(bgr, player_center, prev_gray=None, debug_draw=None):
    """返回 (bullets_xy, grid)。grid 形状 [N_SECTORS*len(RINGS_PX)]。
    prev_gray: 上一帧灰度；给定时敌弹只计"在动"的。"""
    blobs = detect_enemy_bullets(bgr, prev_gray)
    px, py = player_center
    rings = len(RINGS_PX)
    grid = np.zeros(N_SECTORS * rings, dtype=np.float32)
    for (cx, cy, _area) in blobs:
        d = ((cx - px) ** 2 + (cy - py) ** 2) ** 0.5
        if d < 1:
            continue
        # 找所在圈(按RINGS阈值)
        ring = next((ri for ri, rr in enumerate(RINGS_PX) if d <= rr), rings - 1)
        a = np.degrees(np.arctan2(cy - py, cx - px))
        sec = int(((a + 180) % 360) / (360.0 / N_SECTORS)) % N_SECTORS
        # 威胁度：越近越高(内圈系数更大)
        coef = 1.0 / (1.0 + d / RINGS_PX[-1])
        grid[sec * rings + ring] += coef
        if debug_draw is not None:
            cv2.circle(debug_draw, (int(cx), int(cy)), 6, (0, 0, 255), 1)
    return [(b[0], b[1]) for b in blobs], grid


def detect_drops(bgr, debug_draw=None):
    """掉落物(红/蓝果等)。v0 先按色区分，闪烁(is_blinking)留给运行时跨帧。"""
    drops = []
    for name, ranges in DROP_HSV.items():
        mask = _mask_for_ranges(bgr, ranges)
        for (cx, cy, _area) in _small_blobs(mask, DROP_AREA_MIN, DROP_AREA_MAX, 60):
            drops.append({"type": name, "c": (float(cx), float(cy))})
            if debug_draw is not None:
                cv2.circle(debug_draw, (int(cx), int(cy)), 10,
                           (255, 0, 0) if name != "blue" else (255, 200, 0), 2)
    return drops


def grid_sector_angle(sec):
    """扇区中心角(度)供避让/调试用。"""
    return -180.0 + (sec + 0.5) * (360.0 / N_SECTORS)


def obstacle_grid(bgr, player_center, rings=None):
    """障碍物粗栅格(16扇区×N圈)：用边缘密度近似"实体区域"。
    不区分敌人/藤蔓/电锯，只告诉决策层"哪个方向有东西"(YOLO漏检也能兜底)。"""
    rings = rings or RINGS_PX
    px, py = player_center
    h, w = bgr.shape[:2]
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 45, 130)
    k = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    edges = cv2.dilate(edges, k, iterations=1)
    g = np.zeros(N_SECTORS * len(rings), dtype=np.float32)
    ys, xs = np.nonzero(edges)
    if len(xs) == 0:
        return g
    dx = xs - px
    dy = ys - py
    d = np.hypot(dx, dy)
    ang = np.degrees(np.arctan2(dy, dx))
    secs = ((((ang + 180.0) % 360.0) // (360.0 / N_SECTORS)).astype(int)) % N_SECTORS
    # 越近权重越大
    coef = 1.0 / (1.0 + d / rings[-1])
    for ri in range(len(rings)):
        if ri == 0:
            m = d <= rings[0]
        else:
            m = (d > rings[ri - 1]) & (d <= rings[ri])
        if not m.any():
            continue
        for s in range(N_SECTORS):
            mm = m & (secs == s)
            if mm.any():
                g[s * len(rings) + ri] += float(coef[mm].sum())
    return g
