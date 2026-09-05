"""
枪豆人 (Bang Bang Barrage) YOLO v1 类别清单（唯一权威来源）
简单·无限模式可出现的类型；困难专属类按课程 S6 再扩充。

类别 = (英文标识token, 中文名, 出现说明)
- 英文 token：供 data.yaml / 模型内部使用
- labelme 标注时【中文名 或 英文 token 都可以】→ label_to_id() 都能识别
- ⚠️ id 一律由 CLASSES 列表顺序自动派生；别名用 token 键控 → 永不漂移

⚠️ 合并说明（听雨确认）：
- hand_heart 合并 心心果手(红) + 蓝心心果手 + 原地心心手 → 标"心心果手"
- hand_diamond 合并 钻石手 + 乱跑钻石手（奖励段）
- 2026-09-04 已移除 flower_eye 白眼球花（图鉴中不存在，待确认真实来源后加回）
"""
# (token, 中文, 出现说明)
CLASSES = [
    ("player_boat",          "玩家船",            "常态"),
    ("hand_green",           "抓抓绿手",          "常态"),
    ("hand_orange_thumb",    "点赞橙手",          "常态"),
    ("hand_red_peace",       "比耶红手",          "常态"),
    ("hand_purple",          "追追紫手",          "常态"),
    ("hand_yellow_glove",    "黄手套手",          "常态"),
    ("hand_red_glove",       "红手套手",          "常态"),
    ("hand_heart",           "心心果手(红/蓝/原地合并)", "常态"),
    ("hand_blue",            "否定蓝手",          "常态(垂直移动)"),
    ("hand_bracelet",        "钻石手镯手",        "常态·全关卡(掉3钻)"),
    ("hand_diamond",         "钻石手/乱跑钻石手(合并)", "仅Boss奖励段"),
    ("flower_cross",         "十字花(桃红)",      "常态"),
    ("flower_cross_diag",    "斜十字花(明黄)",    "常态"),
    ("flower_lily",          "铃兰",              "常态"),
    ("flower_lily_pink",     "粉铃兰",            "常态"),
    ("flower_bomb",          "炸弹花",            "常态"),
    ("flower_bomb_poison",   "毒炸弹花",          "常态"),
    ("flower_bomb_short",    "短手炸弹花",        "常态"),
    ("flower_bomb_big",      "大炸弹花",          "常态"),
    ("pirate_normal",        "普通海盗(帽/无帽)", "常态"),
    ("pirate_shield_spin",   "转转盾海盗",        "常态"),
    ("pirate_shield",        "持盾海盗",          "常态"),
    ("pirate_cross",         "十字花海盗",        "常态"),
    ("pirate_cross_diag",    "斜十字花海盗",      "常态"),
    ("ghost_plain",          "呆呆幽灵",          "黄昏/夜晚"),
    ("ghost_chaser",         "追追幽灵",          "黄昏/夜晚"),
    ("ghost_hat",            "帽子幽灵",          "不分时段"),
    ("ghost_shield",         "持盾幽灵",          "黄昏/夜晚"),
    ("ghost_diamond",        "钻石幽灵",          "奖励段"),
    ("vine_leaf",            "叶子藤蔓",          "后期生长"),
    ("vine_cross",           "十字花藤蔓",        "后期生长"),
    ("vine_cross_diag",      "斜十字花藤蔓",      "后期生长"),
    ("vine_lily",            "铃兰藤蔓",          "后期生长"),
    ("vine_ghost",           "幽灵藤蔓",          "黄昏/夜晚生长"),
    ("vine_diamond",         "钻石藤蔓",          "奖励段"),
    ("saw_orange",           "橙色电锯",          "中后期事件"),
    ("saw_double",           "双电锯",            "中后期事件"),
    ("saw_diamond",          "钻石电锯",          "中后期事件"),
    ("boss_pirate",          "大海盗",            "Boss"),
    ("boss_pirate_shield",   "持盾大海盗",        "Boss"),
    ("boss_pirate_lily",     "铃兰大海盗",        "Boss"),
    ("boss_pirate_shield_lily", "持盾铃兰大海盗", "Boss"),
    ("boss_ghost",           "大幽灵",            "Boss"),
    ("boss_vine_long",       "长长藤蔓",          "Boss"),
    ("boss_saw_snake",       "电锯蛇",            "Boss"),
]

# 每类额外中文别名（按 token 键控；每类的中文规范名/去括号名会自动加入，无需在此重复）
CN_ALIAS_GROUPS = {
    "player_boat":       ["船", "玩家"],
    "hand_green":        ["抓抓绿", "绿手", "抓抓手"],
    "hand_orange_thumb": ["点赞手", "橙手", "点赞橙"],
    "hand_red_peace":    ["比耶手", "红手", "比耶红", "胜利手"],
    "hand_purple":       ["追追手", "紫手", "追追紫"],
    "hand_yellow_glove": ["黄手套"],
    "hand_red_glove":    ["红手套"],
    "hand_heart":        ["蓝心心果手", "原地心心手", "心心手", "心心果",
                          "红果手", "蓝果手", "红心心手"],
    "hand_blue":         ["蓝手", "否定蓝"],
    "hand_bracelet":     ["手镯手", "钻石手镯"],
    "hand_diamond":      ["乱跑钻石手", "乱跑钻石"],
    "flower_cross":      ["红色花", "红花", "桃红花"],
    "flower_cross_diag": ["黄色花", "黄花", "明黄花"],
    "pirate_normal":     ["海盗", "手枪海盗"],
    "pirate_shield_spin": ["旋转盾海盗"],
    "ghost_plain":       ["普通幽灵", "呆幽灵"],
    "ghost_chaser":      ["追击幽灵"],
    "vine_leaf":         ["藤蔓", "普通藤蔓", "叶藤蔓"],
    "saw_orange":        ["电锯", "橙电锯"],
    "boss_pirate":       [],
    "boss_saw_snake":    ["电锯蛇boss"],
}

# 每类英文别名（按 token 键控）
EN_ALIAS_GROUPS = {
    "player_boat":    ["player", "boat", "ship"],
    "hand_heart":     ["hand_heart_red", "hand_heart_blue", "hand_heart_fruit",
                       "hand_heart_still", "blue_fruit_hand"],
    "flower_lily":    ["lily", "bellflower"],
    "boss_pirate":    ["boss_big_pirate", "big_pirate"],
    "boss_saw_snake": ["chain_saw_snake", "chainsaw_snake"],
}

# ========== 派生映射（只依赖 CLASSES 顺序，别动） ==========
CLASS_NAMES = {i: c[0] for i, c in enumerate(CLASSES)}   # id -> token
CN_NAMES = {i: c[1] for i, c in enumerate(CLASSES)}       # id -> 中文名
TOKEN_TO_ID = {c[0]: i for i, c in enumerate(CLASSES)}    # token -> id
LABEL_MAP = {token: i for i, (token, _, _) in enumerate(CLASSES)}

# 中文 -> id（每类自动含 规范中文名 与 去括号名/斜杠分段名；外加别名）
CN_TO_ID = {}
for i, (token, cn, _) in enumerate(CLASSES):
    CN_TO_ID[cn] = i
    base = cn.split("(")[0].strip()                 # "十字花(桃红)" -> "十字花"
    CN_TO_ID[base] = i
    for part in base.split("/"):                    # "钻石手/乱跑钻石手" -> 两款都认
        part = part.strip()
        if part:
            CN_TO_ID[part] = i
    for alias in CN_ALIAS_GROUPS.get(token, []):
        CN_TO_ID[alias] = i

EN_TO_ID = {alias: i for i, (token, _, _) in enumerate(CLASSES)
            for alias in EN_ALIAS_GROUPS.get(token, [])}


def _norm(label: str) -> str:
    """统一空白（含全角），去首尾。"""
    s = label.replace("\u3000", " ")
    return " ".join(s.split()).strip().lower()


def label_to_id(label: str) -> int:
    """把 labelme 里用户写的标签转成类别 id。
    支持：中文名(含别名/括号名) / 英文token / 英文别名，大小写与空格容错。
    """
    key = _norm(label)

    # 1) 中文
    if key in CN_TO_ID:
        return CN_TO_ID[key]
    # 2) 英文 token / 别名
    if key in TOKEN_TO_ID:
        return TOKEN_TO_ID[key]
    if key in EN_TO_ID:
        return EN_TO_ID[key]
    # 3) 取第一个词再试
    first = key.split()[0] if key else ""
    if first in CN_TO_ID:
        return CN_TO_ID[first]
    if first in TOKEN_TO_ID:
        return TOKEN_TO_ID[first]
    if first in EN_TO_ID:
        return EN_TO_ID[first]

    raise KeyError(f"未知标签: {label}")


if __name__ == "__main__":
    print(f"共 {len(CLASSES)} 类 (id 0~{len(CLASSES)-1})：")
    for i, (token, cn, note) in enumerate(CLASSES):
        print(f"  {i:2d}  {cn:<22} [{token}] {note}")
