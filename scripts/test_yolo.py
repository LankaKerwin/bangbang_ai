"""
枪豆人 YOLO 实时检测预览（小窗口版）
用训练好的模型实时检测游戏画面，画框显示在一个【小窗口】里，
不挡住整个屏幕，方便边玩游戏边看识别效果。

用法：
    python scripts/test_yolo.py
    python scripts/test_yolo.py --model models/yolo/bb_yolov8s/weights/best.pt --win_width 480

参数：
    --model      模型路径，默认 bb_yolov8s/best.pt
    --region     "left,top,width,height" 游戏区域；默认全屏
    --win_width   小窗口宽度像素(高度等比)，默认480（在2K屏上只占左下角一小块）
    --conf       置信度阈值，默认0.35

快捷键：
    q  退出
    w  窗口变宽一点    s  窗口变窄一点
"""
import os
import sys
import time
import argparse

import cv2
import numpy as np
import mss
from ultralytics import YOLO

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import bb_classes  # noqa: E402


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=None, help="模型路径")
    parser.add_argument("--region", default=None, help='"left,top,width,height"，默认全屏')
    parser.add_argument("--win_width", type=int, default=480, help="小窗口宽度(默认480)")
    parser.add_argument("--conf", type=float, default=0.35, help="置信度阈值")
    args = parser.parse_args()

    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if args.model is None:
        args.model = os.path.join(base_dir, "models", "yolo", "bb_yolov8s",
                                  "weights", "best.pt")
    if not os.path.exists(args.model):
        print(f"找不到模型: {args.model}")
        print("先跑 python scripts/train_yolo.py 训练（或用 --model 指定已有权重）")
        return

    model = YOLO(args.model)
    infer_size = 640
    win_width = args.win_width
    win_name = "BB Detect (small)"

    print(f"模型: {args.model}")
    print(f"共 {len(model.names)} 类；小窗口宽 {win_width}px，按 q 退出，w/s 调窗口大小")
    print("提示：中文名对照见 yolo_data/classes_v1.md")

    cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(win_name, win_width, int(win_width * 9 / 16))
    try:
        cv2.setWindowProperty(win_name, cv2.WND_PROP_TOPMOST, 1)  # 置顶
    except Exception:
        pass

    with mss.mss() as sct:
        if args.region:
            l, t, w, h = [int(x) for x in args.region.split(",")]
            monitor = {"left": l, "top": t, "width": w, "height": h}
        else:
            monitor = sct.monitors[1]

        while True:
            t0 = time.time()

            img = np.array(sct.grab(monitor))
            img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)

            # 跨帧跟踪（同目标ID稳定，减少闪烁）
            results = model.track(img, imgsz=infer_size, conf=args.conf, iou=0.5,
                                  persist=True, tracker="bytetrack.yaml", verbose=False)
            annotated = results[0].plot()

            # FPS
            fps = 1.0 / (time.time() - t0)
            cv2.putText(annotated, f"FPS: {fps:.1f}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            # 各类计数（显示前5多的类别，防刷屏）
            boxes = results[0].boxes
            cnt = {}
            if boxes is not None and len(boxes) > 0:
                for c in boxes.cls:
                    cid = int(c)
                    cnt[cid] = cnt.get(cid, 0) + 1
            top = sorted(cnt.items(), key=lambda x: -x[1])[:5]
            text = " ".join(f"{bb_classes.CN_NAMES.get(i, i)}:{n}" for i, n in top)
            cv2.putText(annotated, text or "no target", (10, 70),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

            # 缩到小窗口
            hh, ww = annotated.shape[:2]
            win_w = max(240, min(win_width, ww))
            win_h = int(hh * win_w / ww)
            small = cv2.resize(annotated, (win_w, win_h))

            cv2.imshow(win_name, small)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('w'):
                win_width = min(ww, win_width + 80)
                cv2.resizeWindow(win_name, win_width, int(win_width * win_h / win_w))
            elif key == ord('s'):
                win_width = max(240, win_width - 80)
                cv2.resizeWindow(win_name, win_width, int(win_width * win_h / win_w))

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
