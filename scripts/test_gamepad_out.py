# -*- coding: utf-8 -*-
"""
test_gamepad_out.py — 虚拟手柄通道自动测试(零键鼠交互!)
=========================================================
⚠️ 枪豆人会绑定"开局时活动的输入设备": 碰键盘=绑键盘, 碰鼠标=绑鼠标。
本脚本启动后 8 秒自动开始(给你时间把手离开键鼠), 全程无需任何输入:
    1. 右摇杆右 2s   2. 右摇杆上 2s   3. 左摇杆右 2s
    4. 左摇杆上 2s   5. RT 扳机 2s    6. 复位
步骤间停 2.5s 供你观察游戏反应。
用法: 游戏在正常战斗画面 → python -u scripts/test_gamepad_out.py → 手离键鼠看画面
"""
import time
import vgamepad as vg


def hold(g, apply_fn, sec):
    apply_fn(g)
    t = time.time()
    while time.time() - t < sec:
        g.update()
        time.sleep(0.01)
    g.reset()
    g.update()


def main():
    print("8 秒后自动开始(现在把手离开键盘鼠标!)...")
    time.sleep(8)
    g = vg.VX360Gamepad()
    print("vg 已创建。步骤 1: 右摇杆→右 2s")
    hold(g, lambda gg: gg.right_joystick_float(x_value_float=1.0, y_value_float=0.0), 2)
    time.sleep(2.5)
    print("步骤 2: 右摇杆→上 2s")
    hold(g, lambda gg: gg.right_joystick_float(x_value_float=0.0, y_value_float=1.0), 2)
    time.sleep(2.5)
    print("步骤 3: 左摇杆→右 2s")
    hold(g, lambda gg: gg.left_joystick_float(x_value_float=1.0, y_value_float=0.0), 2)
    time.sleep(2.5)
    print("步骤 4: 左摇杆→上 2s")
    hold(g, lambda gg: gg.left_joystick_float(x_value_float=0.0, y_value_float=1.0), 2)
    time.sleep(2.5)
    print("步骤 5: RT 扳机 2s")
    hold(g, lambda gg: gg.right_trigger_float(value_float=1.0), 2)
    time.sleep(2.5)
    print("步骤 6: 复位完成。请描述你看到的游戏反应。")


if __name__ == "__main__":
    main()
