# M0b 感知状态接口（2026-09-05 收尾 v2：+受伤记忆 +障碍栅格）

入口：`perception.m0b_state(bgr, detections, prev_state, prev_gray, hearts, include_drops, hurt_recent, hurt_from)`
→ 结构化 dict；`perception.m0b_vector(st)` → **固定 159 维 np.float32**。

## 结构化字段
- self: hearts / hurt_recent(0-1衰减) / hurt_from(受伤时朝向线索) / ammo / aim_deg / radius / half
- counts: reg / ghost / hat / saw_small / saw_big
- nearest[≤6]: {dx,dy,dist_norm,ang_norm,grp,inr}
- grid[48]: 弹幕威胁栅格（实时给 prev_gray 只计移动弹）
- obstacle[48]: **障碍物粗栅格**(边缘密度，兜 YOLO 漏检的藤蔓/实体)
- bullets / drops[≤4]（候选级，吃不吃由 RL 学）

## 扁平向量布局（159）
```
[0:  hearts/10, n_enemy/30, hurt_recent, hurt_from]
[4:  ammo(-1未知), aim_norm, radius/2000, half/45]
[8:  reg/20, ghost/20, hat/20, saw_small/20, saw_big/20]        # 13
[13: nearest×6×{ang_norm,dist_norm,grp/4,inr,dx}]               # 30 →43
[43: grid 48]                                                   # 48 →91
[91: obstacle 48]                                               # 48 →139
[139:drops×4×{dx,dy,red,blue,diamond}]                          # 20 →159
```
## 已知取舍(交给 RL 兜底)
- ammo 视觉闪烁 → driver 内部弹药估计(m1-stable 起)
- 掉落识别 v0 候选：碎片/静止怪可能混入 → RL 靠"吃到才加 HP"学会分辨
- 弹幕栅格单帧无移动判据，实时用 prev_gray 差分只计飞行弹
- **障碍栅格是"边缘密度"粗近似**，不能区分具体是啥 → 用于"看不见也能绕开"，精准仍靠 YOLO
- 受伤记忆：血量 ROI 偶发误读，做了"只认 1-3 点且非大跳变"过滤，仍非完美
- 显式档位/头顶 OCR 未做（半张角按半径查表代理；OCR 归 L3）
