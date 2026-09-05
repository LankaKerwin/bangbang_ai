# M0b 感知状态接口（2026-09-05 收尾）

入口：`perception.m0b_state(bgr, detections, prev_state, prev_gray, hearts, include_drops)`
→ 结构化 dict；`perception.m0b_vector(st)` → **固定 109 维 np.float32**。

## 结构化字段
- self: hearts / ammo / aim_deg / radius(攻击距离) / half(半张角)
- counts: reg(普通敌) / ghost / hat(帽子幽灵) / saw_small(可打) / saw_big(事件)
- nearest[≤6]: 每项 {dx,dy(归一), dist_norm(相对半径), ang_norm(0-1), grp(0普通/1幽灵/2帽灵/3小锯/4大锯), inr}
- grid[48]: 16扇区×3圈 弹幕威胁栅格（实时给 prev_gray 时只计**移动**弹）
- bullets(弹数) / drops[≤4]: {type(red/blue/diamond), dx, dy}（**候选级**，吃不吃由 RL 用奖励学）

## 扁平向量布局（109）
```
[0: hearts/10, n_enemy/30,
   ammo(-1未知), aim_norm, radius/2000, half/45,
   reg/20, ghost/20, hat/20, saw_small/20, saw_big/20]           # 11
[11:  nearest×6 × {ang_norm,dist_norm,grp/4,inr,dx}]             # 30
[41:  grid 48]                                                    # 48
[89:  drops×4 × {dx,dy,red,blue,diamond}]                         # 20
合计 109
```
## 已知取舍(交给 RL 兜底)
- ammo 视觉闪烁 → driver 内有"内部弹药估计"(m1-stable 起)
- 掉落识别为 v0 候选：击杀碎片/静止怪可能混入 → RL 靠"吃到才加 HP"奖励学会分辨
- 弹幕栅格单帧无移动判据，实时用 prev_gray 差分只计飞行弹
- 显式档位/头顶数字 OCR 未做（半张角按半径查表代理；OCR 归 L3）
