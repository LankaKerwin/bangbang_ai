# -*- coding: utf-8 -*-
"""
diag_xinput.py — XInput 槽位诊断(实体 vs vg vs 时灵时不灵根因)
================================================================
纯查询 + 创建/销毁一个 vg。不需要游戏、不碰键鼠、无副作用。
输出槽 0-3 占用 + vg 落在哪槽(get_index)。
"""
import ctypes
import time
import gc
import vgamepad as vg


def load_xinput():
    for name in ("xinput1_4", "xinput1_3", "xinput9_1_0"):
        try:
            return ctypes.windll[name]
        except Exception:
            continue
    return None


class _GAMEPAD(ctypes.Structure):
    _fields_ = [("wButtons", ctypes.c_ushort), ("bLeftTrigger", ctypes.c_ubyte),
                ("bRightTrigger", ctypes.c_ubyte), ("sThumbLX", ctypes.c_short),
                ("sThumbLY", ctypes.c_short), ("sThumbRX", ctypes.c_short),
                ("sThumbRY", ctypes.c_short)]


class _STATE(ctypes.Structure):
    _fields_ = [("dwPacketNumber", ctypes.c_ulong), ("Gamepad", _GAMEPAD)]


def scan(xi, tag):
    occ = [i for i in range(4)
           if xi.XInputGetState(i, ctypes.byref(_STATE())) == 0]
    print("  [%s] 槽 0-3 有设备: %s" % (tag, occ if occ else "无"))


def main():
    xi = load_xinput()
    if xi is None:
        print("找不到 XInput dll!")
        return
    print("XInput dll 就绪")
    scan(xi, "创建vg前")
    try:
        g = vg.VX360Gamepad()
        print("vg 创建成功 | get_index = %s" % g.get_index())
    except Exception as e:
        print("vg 创建失败: %s" % e)
        return
    scan(xi, "创建vg后")
    time.sleep(1)
    g.reset()
    g.update()
    print("---")
    print("解读: get_index 有数字=vg占该槽。若 vg 永远在槽1+ 而实体占槽0,")
    print("且游戏只认槽0 → vg 只在实体恰好掉线/让位时才被认(时灵时不灵的根源)。")
    del g
    gc.collect()


if __name__ == "__main__":
    main()
