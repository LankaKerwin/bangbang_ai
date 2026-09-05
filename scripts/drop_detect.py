# -*- coding: utf-8 -*-
"""
drop_detect.py — 掉落物检测(形象模板匹配版)
================================================================
红果/蓝果/钻石用【特写模板 + 多尺度模板匹配】识别（听雨：颜色法太易错）。
蓝果颜色通道可作补充(见 bullet_grid.DROP_HSV)，本模块主打形象。

API:
    drops = detect_drops(bgr, debug_draw=None)
      → [{type:'red'|'blue'|'diamond', c:(x,y), score, w, h}]
模板: assets/templates/drops/{red,blue,diamond}.png
"""
import os
import cv2
import numpy as np

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TPL_DIR = os.path.join(BASE, "assets", "templates", "drops")

SCALES = [1.0, 0.8, 0.6, 0.45, 0.33, 0.25]   # 模板缩放候选(场景中果可能更小)
CONF = 0.50                                          # 匹配阈值(待标定)
NMS_OVERLAP = 0.5                                    # 去重半径=框尺寸*0.5
SEARCH_SCALE = 0.5                                   # 半分辨率搜索(提速~4x)
TOP_K = 40                                           # 每模板每尺度候选上限(先排分再查色)


def _imread_u(path):
    with open(path, "rb") as f:
        return cv2.imdecode(np.frombuffer(f.read(), np.uint8), cv2.IMREAD_COLOR)


def _load_templates():
    tpls = []
    for name, typ in (("red.png", "red"), ("blue.png", "blue"), ("diamond.png", "diamond")):
        p = os.path.join(TPL_DIR, name)
        if os.path.exists(p):
            tpls.append((typ, _imread_u(p)))
    return tpls


def _nms(matches):
    """matches: list of dict {score,x,y,w,h,typ}; 重叠框只留分数最高。"""
    matches = sorted(matches, key=lambda m: -m["score"])
    kept = []
    for m in matches:
        ok = True
        for k in kept:
            cx = abs((m["x"] + m["w"] / 2) - (k["x"] + k["w"] / 2))
            cy = abs((m["y"] + m["h"] / 2) - (k["y"] + k["h"] / 2))
            if cx < NMS_OVERLAP * max(m["w"], k["w"]) and cy < NMS_OVERLAP * max(m["h"], k["h"]):
                ok = False
                break
        if ok:
            kept.append(m)
    return kept


def _blocked_by_big_enemy(cx, cy, area_self, boxes):
    """候选中心在某个【明显更大的】YOLO敌人框内 → 排除(手/怪大, 果小)"""
    if not boxes:
        return False
    for (x1, y1, x2, y2) in boxes:
        if x1 <= cx <= x2 and y1 <= cy <= y2:
            if (x2 - x1) * (y2 - y1) >= area_self * 4.0:
                return True
    return False


def detect_drops(bgr, enemy_boxes=None, debug_draw=None):
    """形象模板匹配掉落物(半分辨率加速)。
    enemy_boxes: YOLO 检出的敌人框(排除，防手类静止怪当果)；左上HUD区固定排除。
    返回 drops 列表；可选 debug_draw 画框。"""
    tpls = _load_templates()
    hh, ww = bgr.shape[:2]
    sh, sw = max(8, hh // 2), max(8, ww // 2)          # 半分辨率
    small = cv2.resize(bgr, (sw, sh))
    small_gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
    inv = 1.0 / SEARCH_SCALE

    cands = []
    for typ, tpl in tpls:
        th, tw = tpl.shape[:2]
        tpl_gray = cv2.cvtColor(tpl, cv2.COLOR_BGR2GRAY)
        for s in SCALES:
            if min(th, tw) * s * SEARCH_SCALE < 8:
                continue
            nw = max(4, int(tw * s * SEARCH_SCALE))
            nh = max(4, int(th * s * SEARCH_SCALE))
            if nh > sh or nw > sw:
                continue
            ts = cv2.resize(tpl, (nw, nh))
            tg = cv2.resize(tpl_gray, (nw, nh))
            res = cv2.matchTemplate(small_gray, tg, cv2.TM_CCOEFF_NORMED)
            locs = np.where(res > CONF)
            scored = sorted(zip(locs[1], locs[0]), key=lambda p: -res[p[1], p[0]])
            for x, y in scored[:TOP_K]:
                # 颜色复核：区域均值与模板均值差距小才算
                roi = small[y:y + nh, x:x + nw]
                if roi.shape[:2] != (nh, nw):
                    continue
                tm = cv2.resize(tpl, (nw, nh))
                diff = float(np.mean(np.abs(roi.astype(np.float32) - tm.astype(np.float32))))
                if diff > 45:
                    continue
                cands.append({"score": float(res[y, x]), "x": int(x * inv),
                              "y": int(y * inv), "w": int(nw * inv), "h": int(nh * inv),
                              "typ": typ})
    kept = _nms(cands)
    hh0, ww0 = bgr.shape[:2]
    hud_zone = (0, 0, ww0 * 0.24, hh0 * 0.11)   # 左上血量/蓝星 HUD
    drops = []
    for k in kept:
        cx = k["x"] + k["w"] / 2
        cy = k["y"] + k["h"] / 2
        in_hud = (cx < hud_zone[0] or cy < hud_zone[1]
                  or cx > hud_zone[2] or cy > hud_zone[3])
        blocked = _blocked_by_big_enemy(cx, cy, k["w"] * k["h"], enemy_boxes)
        if in_hud or blocked:
            continue
        drops.append({"type": k["typ"], "c": (float(cx), float(cy)),
                      "score": k["score"], "w": k["w"], "h": k["h"]})
        if debug_draw is not None:
            color = (0, 255, 0) if k["typ"] == "red" else ((255, 180, 0) if k["typ"] == "blue" else (0, 200, 255))
            cv2.rectangle(debug_draw, (k["x"], k["y"]),
                          (k["x"] + k["w"], k["y"] + k["h"]), color, 2)
            cv2.putText(debug_draw, k["typ"], (k["x"], k["y"] - 6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    return drops
