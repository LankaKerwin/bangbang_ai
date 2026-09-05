# 🎯 枪豆人 (Bang Bang Barrage) AI — 项目

> ⚠️ **声明：本项目仍在开发中，尚未完善**（当前进度：YOLO v1 感知可用、M1 雏形自动循环跑通，战斗策略/RL 仍在推进）。代码与配置随时会变动，**请勿随意下载使用**，仅供参考学习。

用 **YOLO 感知 + OCR + RL(PPO)** 训练 AI 玩《枪豆人》。当前处于 **M0a 感知数据工程完成 → 推进 M0b 感知 / M1 教练** 阶段。

> 设计文档：
> - `docs/plan_overview.md` — 完整方案综述（感知/状态/动作/RL/课程/里程碑）
> - `docs/enemy_behavior_table.md` — 敌人/Boss 类型-行为知识库（RL 先验来源）

## 当前进度
- [x] 方案定稿：角色=小草 + 武器=霰弹枪 + 简单·无限；算法=PPO；预热=脚本教练(BC→PPO)
- [x] YOLO v1 类别清单（46 类）→ `yolo_data/classes_v1.md` / `scripts/bb_classes.py`
- [x] M0a 工具链骨架（采集/筛选/标注转换/训练/实时预览）
- [ ] 录制素材并标注 → 训练 YOLO v1
- [ ] M0b 弹幕栅格 + OCR + 状态融合
- [ ] M1 脚本教练 → M2 BC → M3 PPO（见 plan_overview.md 里程碑）

## M0a 工具链使用顺序

```bash
cd AI train/bangbang_ai

# 1) 采集实战截图（玩游戏时跑，可同时录不同时段）
python scripts/capture_yolo_frames.py --max 800

# 2) 有初版模型后：预筛选"有敌人"的图，减少标注量
python scripts/filter_yolo_frames.py --input yolo_data/images/to_label --model <best.pt>

# 3) 用 labelme 标注 yolo_data/images/to_label/has_target 里的图
#    标签填英文 token（对照 yolo_data/classes_v1.md）

# 4) 自动预标注（可选：模型够强时先让模型框，人只改）
python scripts/auto_label_yolo.py --dir yolo_data/images/to_label/has_target --model <best.pt>

# 5) labelme JSON -> YOLO txt + 训练/验证切分
python scripts/labelme_to_yolo.py

# 6) 训练
python scripts/train_yolo.py

# 7) 实时预览（小窗口，边玩边看识别效果；w/s 调宽窄，q 退出）
python scripts/test_yolo.py --model models/yolo/bb_yolov8s/weights/best.pt
```

## 依赖
```
ultralytics        # YOLOv8（tlou2 环境已有）
opencv-python mss  # 截屏/显示
numpy
```
> 模型预训练权重优先复用 `D:\AI train\tlou2_ai\yolov8s.pt`（train_yolo.py 已处理）。

## 目录结构
```
bangbang_ai/
├── docs/                    # 方案 & 行为知识库
├── scripts/
│   ├── bb_classes.py        # 类别清单（唯一权威）
│   ├── capture_yolo_frames.py
│   ├── filter_yolo_frames.py
│   ├── auto_label_yolo.py
│   ├── labelme_to_yolo.py
│   ├── train_yolo.py
│   └── test_yolo.py         # 小窗口实时预览
└── yolo_data/
    ├── data.yaml
    ├── classes_v1.md        # 中文对照/标注指引
    ├── images/{train,val,to_label/...}
    └── labels/{train,val}
```
