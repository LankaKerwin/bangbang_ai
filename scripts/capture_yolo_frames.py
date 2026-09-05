"""
枪豆人 YOLO 数据采集（实战截图版）
游戏画面实时截屏存高清jpg，用于YOLO数据集。
跟推理时YOLO看到的画质一致，避免训练-推理分布偏移。

用法：
    python scripts/capture_yolo_frames.py --output yolo_data/images/to_label --interval 0.5 --max 1000

参数：
    --output    输出目录，默认 yolo_data/images/to_label
    --interval  每隔多少秒截一张，默认0.5
    --max       最多截多少张，默认500
    --width     保存宽度(等比)，默认1280
    --region    "left,top,width,height" 可选；默认全屏主显示器
"""
import cv2
import numpy as np
import mss
import time
import os
import argparse


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default=None, help="输出目录")
    parser.add_argument("--interval", type=float, default=0.5, help="截屏间隔(秒)")
    parser.add_argument("--max", type=int, default=500, help="最多截多少张")
    parser.add_argument("--width", type=int, default=1280, help="保存宽度(高度等比)")
    parser.add_argument("--region", default=None, help='"left,top,width,height"，默认全屏')
    args = parser.parse_args()

    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if args.output is None:
        args.output = os.path.join(base_dir, "yolo_data", "images", "to_label")
    os.makedirs(args.output, exist_ok=True)

    with mss.mss() as sct:
        if args.region:
            l, t, w, h = [int(x) for x in args.region.split(",")]
            monitor = {"left": l, "top": t, "width": w, "height": h}
        else:
            monitor = sct.monitors[1]  # 主显示器全屏
            l, t, w, h = monitor["left"], monitor["top"], monitor["width"], monitor["height"]

        print("枪豆人 YOLO 数据采集已启动")
        print(f"截屏区域: left={l} top={t} w={w} h={h}")
        print(f"输出目录: {args.output}")
        print(f"间隔: {args.interval}s（约{1/args.interval:.1f}fps），上限 {args.max} 张")
        print("=" * 50)
        print("  启动即开始截图，进游戏战斗画面即可")
        print("  要停止：直接关掉这个命令行窗口")
        print("  提示：录不同时段(黄昏/夜晚)与不同敌人种类才有多样性")
        print("=" * 50)
        time.sleep(2)

        count = 0
        last_time = 0
        while count < args.max:
            now = time.time()
            if now - last_time < args.interval:
                time.sleep(0.01)
                continue
            last_time = now

            img = np.array(sct.grab(monitor))
            img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)

            # 等比缩到目标宽度
            scale = args.width / img.shape[1]
            new_h = int(img.shape[0] * scale)
            img_resized = cv2.resize(img, (args.width, new_h), interpolation=cv2.INTER_AREA)

            fname = f"bb_{int(time.time() * 1000)}.jpg"
            cv2.imwrite(os.path.join(args.output, fname), img_resized,
                        [cv2.IMWRITE_JPEG_QUALITY, 95])
            count += 1
            if count % 20 == 0:
                print(f"  已截取 {count}/{args.max} 张")

    print(f"\n完成！共 {count} 张 → {args.output}")
    print("下一步：filter_yolo_frames.py 预筛选 → labelme 人工标注")


if __name__ == "__main__":
    main()
