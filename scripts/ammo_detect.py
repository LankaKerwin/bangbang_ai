# -*- coding: utf-8 -*-
"""
弹药/装填状态检测 v0 (霰弹枪)
=================================
原理：霰弹枪瞄准时，扇形外缘黄弧上有 1-3 个小黄点 = 当前弹药数。
     通过 HSV 黄色小目标 + 径向聚类（各点距玩家约等半径、角跨 <120°）识别。

用法(库函数)：
    from ammo_detect import detect_ammo
    info = detect_ammo(bgr_frame, player_center=(px, py))
    # info = {ammo:int, radius:float|None, angle_deg:(a0,a1)|None, dots:[(x,y),...]}

调试/测试见 test_ammo.py（可与已知截图真值对拍，输出标注图）。
"""
import cv2
import numpy as np

# ===== 可调参数(用标定截图迭代) =====
YELLOW_H_MIN, YELLOW_H_MAX = 18, 42      # 黄色色相带
YELLOW_S_MIN, YELLOW_V_MIN = 110, 150    # 饱和/明度下限
BLOB_AREA_MIN, BLOB_AREA_MAX = 90, 170   # 真弹药点≈12x12面积120-130；过小噪声/过大图标都滤掉
RADIAL_BIN = 70                            # 半径分箱宽(px)，同一弧上的点距玩家相近
RADIAL_TOL = 100                           # prev_radius 先验搜索容差(px)
ANGLE_SPREAD_MAX = 140.0                   # 弧上各点角跨上限(霰弹扇形通常<120°)
MIN_RAD_FRAC = 0.10                         # 排除玩家身边黄色发光(太近的忽略,占屏高比例)
# =====================================


def _yellow_mask(hsv):
    m1 = (hsv[:, :, 0] >= YELLOW_H_MIN) & (hsv[:, :, 0] <= YELLOW_H_MAX)
    m2 = hsv[:, :, 1] >= YELLOW_S_MIN
    m3 = hsv[:, :, 2] >= YELLOW_V_MIN
    mask = (m1 & m2 & m3).astype(np.uint8) * 255
    k = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k)
    return mask


def detect_ammo(bgr, player_center, prev_radius=None, debug_draw=None):
    """返回 {ammo, radius, angle_deg, dots}。
    prev_radius: 上一帧弹药弧半径；给定时优先在它±RADIAL_TOL内找(时序最稳)。
    debug_draw: 可选BGR图，会在上面画点返回。"""
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    mask = _yellow_mask(hsv)
    n, lab, stats, cents = cv2.connectedComponentsWithStats(mask, 8)

    px, py = player_center
    blobs = []
    for i in range(1, n):
        x, y, w, h, area = stats[i]
        if area < BLOB_AREA_MIN or area > BLOB_AREA_MAX:
            continue
        if w < 8 or h < 8:
            continue
        if not (0.6 < w / max(h, 1) < 1.7):
            continue
        cx, cy = cents[i]
        d = ((cx - px) ** 2 + (cy - py) ** 2) ** 0.5
        a = float(np.degrees(np.arctan2(cy - py, cx - px)))
        blobs.append({"c": (float(cx), float(cy)), "d": d, "a": a, "area": float(area)})

    out = {"ammo": 0, "radius": None, "angle_deg": None, "dots": []}
    if not blobs:
        return out

    h, w = bgr.shape[:2]
    min_rad = h * MIN_RAD_FRAC
    far = [b for b in blobs if b["d"] >= min_rad]
    if not far:
        return out

    # 径向分箱；有 prev_radius 先验则优先在它±RADIAL_TOL内找，否则从近到远找最近紧密簇
    ds = np.array([b["d"] for b in far])
    bins = int((ds.max() - min_rad) // RADIAL_BIN) + 1
    hist, edges = np.histogram(ds, bins=bins, range=(min_rad, ds.max() + 1))

    order = list(range(bins))
    if prev_radius is not None:
        cand = []
        for i in range(bins):
            mid = (edges[i] + edges[i + 1]) / 2
            cand.append((abs(mid - prev_radius), i))
        cand.sort()
        prio = [i for _, i in cand if abs((edges[i] + edges[i + 1]) / 2 - prev_radius) <= RADIAL_TOL]
        order = prio + [i for i in order if i not in prio]

    chosen = None
    for bi in order:
        if hist[bi] < 1:
            continue
        r_lo, r_hi = edges[bi], edges[bi + 1]
        on_arc = [b for b in far if r_lo <= b["d"] < r_hi]
        if len(on_arc) > 3:          # 弹药最多3颗
            continue
        angs = sorted(b["a"] for b in on_arc)
        if angs[-1] - angs[0] > ANGLE_SPREAD_MAX:   # 角跨太大 = 不是扇形上的点
            continue
        chosen = (on_arc, float((r_lo + r_hi) / 2), angs)
        break   # 最近的合格簇就是弹药弧

    if chosen is None:
        return out
    on_arc, radius, angs = chosen
    out["dots"] = on_arc
    out["ammo"] = min(len(on_arc), 3)
    out["radius"] = radius
    out["angle_deg"] = (float(angs[0]), float(angs[-1]))

    if debug_draw is not None:
        cv2.circle(debug_draw, (int(px), int(py)), 6, (255, 0, 255), -1)
        for b in on_arc:
            cv2.circle(debug_draw, (int(b["c"][0]), int(b["c"][1])), 8, (0, 255, 0), 2)
        cv2.putText(debug_draw, f"ammo={out['ammo']}", (10, 110),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
    return out
