# 交接方案：输入层从 vgamepad 切换到合成鼠标+键盘（2026-09-07 归档）

> 状态：**已归档（听雨开学暂停）**。系统其余部分全部就绪，只剩输入层切换 + 验证。
> 回来照本文件做，预计 1-2 小时可接上。
> 相关记忆：`记忆\所有对话\主对话\recent_memory\project\bangbang_ai.md`（2026-09-07 会话归档段）。

## 为什么换

- vgamepad(ViGEm 虚拟 X360 手柄)对游戏绑定**间歇失效**（时灵时不灵），卡死一切训练价值。
- 听雨实测定案（键鼠操作方式）：
  - 瞄准 = **鼠标移动**（键鼠模式下准星跟鼠标走）
  - 开枪 = **左键**，**每点一下开一枪**（所有武器一致，半自动）
  - **空格 = 跳 = 手柄A = 确认**；键盘 A 无效；重开界面键鼠/手柄命令都收
  - 开枪有**后坐力**，把船往相反方向推（开枪=位移，与霰弹大后坐力设定一致）
  - 全程只需要鼠标+左键+空格（重开），**不需要 WASD/方向键**
- 换合成鼠标后 = 走真实键鼠设备路径，绑定问题从根上消失，vg 彻底退役。

## 输入层设计（改 rl_env.py，动作空间不变）

动作空间仍是 `Box(-1,1,3) = [aim_x, aim_y, fire]`（连续瞄准 + 开火意图，**不动训练/模型**）：

| 动作 | 原实现 (vg) | 新实现 (ctypes 合成输入) |
|---|---|---|
| 瞄准 aim_x/aim_y | `gp.left_joystick_float(x, y)` 每步一次 | 每步 `mouse_event(MOUSEEVENTF_MOVE, dx, dy)` **相对移动**，幅度=向量长度×每步像素系数 |
| 开枪 fire>0 | `_fire()`：RT 脉冲 0.08s | 左键 **单击**（按下→抬起），由 FIRE_PERIOD_SEC 纪律节流 |
| 重开(死亡) | vg 手柄 A | **空格** 键按下(0.15s)；菜单确认语义等价 |
| 商店跳过 | vg B+A | 未定（店在键鼠下若同样收键鼠命令 → 用空格确认 + 需查"返回"键；回游戏后补测） |

实现建议：新增 `scripts/input_mouse.py`（模块级函数，仿 `reset_controls(gp)` 接口），
`rl_env.py` 里把 `self.gp` 的调用点替换成鼠标函数；也可保留 `input_mode="mouse"|"vg"` 开关便于对照调试。
注意把 `vg.VX360Gamepad()` 的创建/close 一并去掉（vg 无 delete 方法，现有 close 已是 reset+update+sct.close）。

**后坐力补偿预案**：霰弹大后坐力会把船反推 → 开火后准星相对船会偏；
先裸跑让 PPO 自己学"开完枪回正"，若长时间不学再加每步显式补偿。
（听雨 demo 里每次开火只有 1 帧 ≈ 0.17s，后坐力主要影响的是下一帧的瞄准基线，奖励信号够强时 PPO 能自己学会。）

**像素系数待实测**：aim 向量长度 1 ≈ 每步应移动多少像素 → 在真机上试跑，
观察准星是否"够得着"敌人（短稳验证时用 score 和 fire 命中率校准）。

## 待补测的未知点（回来第一个真机 session 解决）

1. 键鼠下**商店界面**怎么退出（B+A 语义 → 哪个键）？先用空格试"确认"，不行再查设置键位。
2. 合成鼠标相对移动的**像素系数**（见上）。
3. **gentle stop**（`self._last_boat` 门控，船消失就停输出）实机验证 —— 输入修好后跑 3-5 局确认：
   新回合能动、死亡不卡重启、无"6步/37.3s/分=-1/hp=0"死亡循环。

## 验证流程（铁律：先短稳，再长跑）

1. `python scripts/verify_bc.py --zip models/bc/bc_ppo_nofs_open.zip`（模型没换，链路没坏即可跳过）
2. 短稳跑：`python scripts/train_ppo.py --no_fire_since --total_timesteps 4096`（≈25 min，`--resume` 可从存档续）
   - 判稳标准：多局死亡重启正常、score_log.csv 出现 60-420 的正常分、无 6 步死亡循环
3. **通过后才允许**过夜 65000（≈6.5h，22:30-06:30 档）：`python scripts/train_ppo.py --no_fire_since --total_timesteps 65000 --resume`
   - 每 2048 步自动存档 `models/rl/bb_ppo_ckpt_*_steps.zip`；回合成绩进 `models/rl/score_log.csv`

## 关键代码位置（rl_env.py）

- `reset_controls(gp)` L95 / `skip_shop(gp)` L104 / `_fire()` L239(RT脉冲) /
  `_restart_round()` L272(vg A 长按+验证循环) / `step()` L342(L355 左摇杆, L362 fire) /
  `__init__` L141 `self.gp = vg.VX360Gamepad()` / `close()` L460
- 死亡重启状态机逻辑**本身稳定勿动**（输入通道换掉即可，验证循环照用）

## 现状文件（勿删）

- BC 模型：`models/bc/bc_ppo_nofs_open.zip`（bias 0.4：意图 14.9%/灵敏度 45%/误火 12.5%）
- RL 存档：`models/rl/bb_ppo_ckpt_2048/4096/6144_steps.zip`、`bangbang_ppo.zip`
  ⚠️ 6144 存档大部分训练在死亡循环数据上，只作 resume 冷启动基线，别期望策略质量
- 诊断工具：`diag_xinput.py`(槽位) / `test_gamepad_out.py`(纯输出通道自测) / `score_log.csv`
