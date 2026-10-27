# -*- coding: utf-8 -*-
"""
下载困惑度模型（可选）

用法：
    python download_model.py                  # 默认 Qwen2.5-0.5B-Instruct
    python download_model.py --dir models/qwen

说明：
    装了模型，检测会多一个「语言模型困惑度」信号——它衡量这段文字
    对语言模型来说有多"顺"，是目前对改写最鲁棒的指标。
    不装也能用，只是少了这一项。

    模型约 950 MB，从 ModelScope 下载（国内可直连）。
    下载后放在 ./models/ 下，程序会自动发现。
"""
from __future__ import annotations

import argparse
import os
import sys

FILES = [
    "config.json",
    "generation_config.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "vocab.json",
    "merges.txt",
    "model.safetensors",
]

REPO = "Qwen/Qwen2.5-0.5B-Instruct"
MIRRORS = [
    "https://www.modelscope.cn/api/v1/models/{repo}/repo?Revision=master&FilePath={f}",
    "https://hf-mirror.com/{repo}/resolve/main/{f}",
]


def human(n):
    for u in ("B", "KB", "MB", "GB"):
        if n < 1024:
            return f"{n:.1f}{u}"
        n /= 1024
    return f"{n:.1f}TB"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default=os.path.join(os.path.dirname(__file__), "models", "qwen2.5-0.5b"))
    ap.add_argument("--repo", default=REPO)
    args = ap.parse_args()

    os.makedirs(args.dir, exist_ok=True)
    print(f"目标目录：{args.dir}")
    print(f"模型：{args.repo}\n")

    import urllib.request

    for f in FILES:
        dest = os.path.join(args.dir, f)
        if os.path.exists(dest) and os.path.getsize(dest) > 1024:
            print(f"  跳过 {f}（已存在）")
            continue

        ok = False
        for tpl in MIRRORS:
            url = tpl.format(repo=args.repo, f=f)
            try:
                print(f"  下载 {f} … ", end="", flush=True)
                req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
                with urllib.request.urlopen(req, timeout=60) as r, open(dest, "wb") as out:
                    total = int(r.headers.get("Content-Length") or 0)
                    done = 0
                    while True:
                        chunk = r.read(1 << 20)
                        if not chunk:
                            break
                        out.write(chunk)
                        done += len(chunk)
                        if total and done % (20 << 20) < (1 << 20):
                            print(f"\r  {f}: {human(done)}/{human(total)}", end="", flush=True)
                size = os.path.getsize(dest)
                print(f"\r  {f}: {human(size)} ✓")
                ok = True
                break
            except Exception as e:
                print(f"\r  {f}: 失败（{type(e).__name__}）")
                if os.path.exists(dest):
                    os.remove(dest)
        if not ok:
            print(f"\n下载 {f} 失败，请检查网络后重试。")
            sys.exit(1)

    print(f"\n完成。模型位于：{args.dir}")
    print("现在运行检测，会自动启用「困惑度」信号。")


if __name__ == "__main__":
    main()
