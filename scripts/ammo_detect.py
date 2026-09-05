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


def half_angle_for_radius(r):
    """由弹药弧半径(≈攻击距离)推断档位半张角(度)。人工验收定标 2026-09-05。
    白/蓝 r<=775 → 15.5°；黄 r<=855 → 27°；红 >855 → 28.5°"""
    if r is None:
        return None
    if r <= 775.0:
        return 15.5
    if r <= 855.0:
        return 27.0
    return 28.5


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


def detect_arc_ends(bgr, player_center, prev_radius=None, aim_deg=None):
    """找扇形黄弧两端的白色 × 十字标记 → 返回扇形边界角度。
    思路：白色小十字位于弧半径环上、且在弹药点两侧最外端。
    返回 {ends:(a_lo,a_hi)或None, half:半张角, radius, candidates:[(ang,dist,size),...]}
    """
    h, w = bgr.shape[:2]
    info = detect_ammo(bgr, player_center, prev_radius=prev_radius)
    r0 = info["radius"]
    out = {"ends": None, "half": None, "radius": r0, "candidates": []}
    if r0 is None:
        return out

    px, py = player_center
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    sat = hsv[:, :, 1]
    white = ((gray > 185) & (sat < 80)).astype(np.uint8) * 255
    k = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    white = cv2.morphologyEx(white, cv2.MORPH_OPEN, k)
    n, lab, stats, cents = cv2.connectedComponentsWithStats(white, 8)

    ring_lo, ring_hi = r0 - 80, r0 + 80
    cands = []
    for i in range(1, n):
        x, y, bw, bh, area = stats[i]
        if area < 20 or area > 900:
            continue
        if bw > 40 or bh > 40:
            continue
        cx, cy = cents[i]
        d = ((cx - px) ** 2 + (cy - py) ** 2) ** 0.5
        if not (ring_lo <= d <= ring_hi):
            continue
        a = float(np.degrees(np.arctan2(cy - py, cx - px)))
        cands.append({"a": a, "d": d, "s": float(area)})
    out["candidates"] = cands
    if not cands:
        return out

    # 目标 aim：弹药点中点或传入
    if aim_deg is None and info["angle_deg"] is not None:
        angs = [b["a"] for b in info["dots"]]
        aim_deg = float(np.degrees(np.arctan2(np.mean(np.sin(np.radians(angs))),
                                              np.mean(np.cos(np.radians(angs))))))

    def diff(a):
        d = a - (aim_deg if aim_deg is not None else 0.0)
        return (d + 180.0) % 360.0 - 180.0

    # 取白色候选里 与 aim 夹角符号相反、|diff| 最大的一对(限制 -75..-8 与 8..75)
    lefts = [c for c in cands if -75 <= diff(c["a"]) <= -8]
    rights = [c for c in cands if 8 <= diff(c["a"]) <= 75]
    if not lefts or not rights:
        return out
    a_lo = min(lefts, key=lambda c: abs(diff(c["a"]) - (-45)))["a"]
    a_hi = min(rights, key=lambda c: abs(diff(c["a"]) - 45))["a"]
    out["ends"] = (float(a_lo), float(a_hi))
    half = (diff(a_hi) - diff(a_lo)) / 2.0
    out["half"] = float(half)
    return out
