# -*- coding: utf-8 -*-
"""
项目快照脚本 — 把 bangbang_ai 的源码/文档/配置打包成 zip（版本回退用）
用法:
    python scripts/make_snapshot.py "m1-loop-working"        # 打标签快照
输出: D:\AI\AI train\backups\bangbang_ai_<标签>_<日期>.zip
排除: 素材图/模型权重/runs/缓存（体积太大，另行用副本管理）
"""
import os
import sys
import zipfile
from datetime import datetime

BASE = r"D:\AI\AI train\bangbang_ai"
OUT_DIR = r"D:\AI\AI train\backups"

# 打包的顶层内容: 目录名 -> 是否递归整个目录
INCLUDE_DIRS = ["scripts", "docs", "assets"]
INCLUDE_FILES = ["README.md"]
INCLUDE_ROOT_YOLO = ["data.yaml", "classes_v1.md", "label_cheatsheet.txt"]  # yolo_data 下

# 排除(相对路径含此片段即跳过)
SKIP_PARTS = [
    "__pycache__", ".pyc", ".cache",
    "yolo_data/images", "yolo_data/labels",
    "models/", "runs/", "data/raw",
]


def skip(path):
    p = path.replace("\\", "/")
    return any(s in p for s in SKIP_PARTS)


def add_files(zf, root_dir, rel_prefix=""):
    for dirpath, dirnames, filenames in os.walk(root_dir):
        for d in list(dirnames):
            full = os.path.join(dirpath, d)
            if skip(full):
                dirnames.remove(d)
        for fn in filenames:
            full = os.path.join(dirpath, fn)
            if skip(full):
                continue
            rel = os.path.relpath(full, BASE)
            zf.write(full, os.path.join("bangbang_ai", rel))


def main():
    tag = sys.argv[1] if len(sys.argv) > 1 else "snapshot"
    os.makedirs(OUT_DIR, exist_ok=True)
    date = datetime.now().strftime("%Y%m%d_%H%M%S")
    out = os.path.join(OUT_DIR, f"bangbang_ai_{tag}_{date}.zip")

    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as zf:
        for d in INCLUDE_DIRS:
            p = os.path.join(BASE, d)
            if os.path.isdir(p):
                add_files(zf, p)
        for fn in INCLUDE_FILES:
            p = os.path.join(BASE, fn)
            if os.path.isfile(p):
                zf.write(p, os.path.join("bangbang_ai", fn))
        yd = os.path.join(BASE, "yolo_data")
        for fn in INCLUDE_ROOT_YOLO:
            p = os.path.join(yd, fn)
            if os.path.isfile(p):
                zf.write(p, os.path.join("bangbang_ai", "yolo_data", fn))

    size = os.path.getsize(out) / 1024 / 1024
    print(f"快照已生成: {out}  ({size:.1f} MB)")
    print("回退方法: 解压覆盖回 bangbang_ai 即可")
    print("说明: 素材图/模型权重不在快照内，模型副本见 models\\release\\")


if __name__ == "__main__":
    main()
