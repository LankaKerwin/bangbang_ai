# v1 YOLO 类别清单（45 类）— 中文对照 & 标注指引

> labelme 里用的标签 = 中文名或英文标识(token)，见下表。
> 权威来源：`../scripts/bb_classes.py`（增删类别改那一处，再同步 data.yaml 并重跑 labelme_to_yolo.py）
> 2026-09-04：**flower_eye 白眼球花已移除**（图鉴中不存在，待确认其真实来源后按名加回）

| id | 英文标识(token) | 中文 | 出现 | 标注提示 |
|---|---|---|---|---|
| 0 | player_boat | 玩家船 | 常态 | 主角木筏，每帧必有 |
| 1 | hand_green | 抓抓绿手 | 常态 | |
| 2 | hand_orange_thumb | 点赞橙手 | 常态 | 竖大拇指 |
| 3 | hand_red_peace | 比耶红手 | 常态 | V字手势 |
| 4 | hand_purple | 追追紫手 | 常态 | |
| 5 | hand_yellow_glove | 黄手套手 | 常态 | 戴黄手套 |
| 6 | hand_red_glove | 红手套手 | 常态 | 红套黄双层 |
| 7 | hand_heart | 心心果手(红/蓝/原地合并) | 常态 | **红果手/蓝果手/原地心心手 都标这个** |
| 8 | hand_blue | 否定蓝手 | 常态 | 垂直移动、成群 |
| 9 | hand_bracelet | 钻石手镯手 | 全关卡 | 飞快乱跑、掉3钻 |
| 10 | hand_diamond | 钻石手/乱跑钻石手(合并) | 仅奖励段 | |
| 11 | flower_cross | 十字花(桃红) | 常态 | 4向爆喷 |
| 12 | flower_cross_diag | 斜十字花(明黄) | 常态 | 斜4向 |
| 13 | flower_lily | 铃兰 | 常态 | 面朝玩家 |
| 14 | flower_lily_pink | 粉铃兰 | 常态 | 连发3弹、有眼珠 |
| 15 | flower_bomb | 炸弹花 | 常态 | 扔头炸弹 |
| 16 | flower_bomb_poison | 毒炸弹花 | 常态 | 留毒圈 |
| 17 | flower_bomb_short | 短手炸弹花 | 常态 | 原地爆 |
| 18 | flower_bomb_big | 大炸弹花 | 常态 | 极大爆炸 |
| 19 | pirate_normal | 普通海盗(帽/无帽) | 常态 | 手枪 |
| 20 | pirate_shield_spin | 转转盾海盗 | 常态 | 旋盾 |
| 21 | pirate_shield | 持盾海盗 | 常态 | 定向盾 |
| 22 | pirate_cross | 十字花海盗 | 常态 | 头戴十字花 |
| 23 | pirate_cross_diag | 斜十字花海盗 | 常态 | 头戴斜十字花 |
| 24 | ghost_plain | 呆呆幽灵 | 黄昏/夜晚 | 物理免疫、光杀 |
| 25 | ghost_chaser | 追追幽灵 | 黄昏/夜晚 | |
| 26 | ghost_hat | 帽子幽灵 | 不分时段 | 需先打帽 |
| 27 | ghost_shield | 持盾幽灵 | 黄昏/夜晚 | 盾挡光 |
| 28 | ghost_diamond | 钻石幽灵 | 奖励段 | 掉钻石 |
| 29 | vine_leaf | 叶子藤蔓 | 后期 | 边缘生长 |
| 30 | vine_cross | 十字花藤蔓 | 后期 | 头=十字花 |
| 31 | vine_cross_diag | 斜十字花藤蔓 | 后期 | 头=斜十字花 |
| 32 | vine_lily | 铃兰藤蔓 | 后期 | 头=铃兰 |
| 33 | vine_ghost | 幽灵藤蔓 | 黄昏/夜晚 | 光照后退 |
| 34 | vine_diamond | 钻石藤蔓 | 奖励段 | 掉钻石 |
| 35 | saw_orange | 橙色电锯 | 中后期 | 被击冲刺 |
| 36 | saw_double | 双电锯 | 中后期 | 绳连两锯 |
| 37 | saw_diamond | 钻石电锯 | 中后期 | 掉碎钻 |
| 38 | boss_pirate | 大海盗 | Boss | 多帽护甲 |
| 39 | boss_pirate_shield | 持盾大海盗 | Boss | 盾转上次受击方向 |
| 40 | boss_pirate_lily | 铃兰大海盗 | Boss | 连发3弹 |
| 41 | boss_pirate_shield_lily | 持盾铃兰大海盗 | Boss | 上两者结合 |
| 42 | boss_ghost | 大幽灵 | Boss | 3鬼火+盾 |
| 43 | boss_vine_long | 长长藤蔓 | Boss | 六角炸弹头 |
| 44 | boss_saw_snake | 电锯蛇 | Boss | 3/5节 |

## 标注规范
- 每类目标建议 ≥10-20 框；只框"活物"，掉落果/钻石/弹幕不框。
- 框用 rectangle，标签可中文可英文。
- 合并口径：心心果手=红/蓝/原地；钻石手=静止/乱跑（奖励段）。
- 困难专属怪与克苏鲁不标。
