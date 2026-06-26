#!/usr/bin/env python3
"""批量下载 Kaggle PTCG AI Battle 的 episode 回放 JSON。

数据来源：官方每天把 top 局打包成数据集 `kaggle/pokemon-tcg-ai-battle-episodes-<date>`，
`kaggle/pokemon-tcg-ai-battle-episodes-index` 的 manifest.csv 列出有哪些天。
本脚本用 Kaggle 官方 API 按需拉取（已按平均评分筛过，都是较强的局）。

准备：先设置 token（会话级即可）
  export KAGGLE_API_TOKEN=KGAT_xxx        # Windows PowerShell: $env:KAGGLE_API_TOKEN="KGAT_xxx"

刷新可用日期清单（写 data/episodes-index/manifest.csv）：
  kaggle datasets download kaggle/pokemon-tcg-ai-battle-episodes-index -p data/episodes-index --unzip

用法示例：
  # 最近 1 天里取 300 局（默认行为，控制磁盘）
  python tools/download_replays.py --latest 1 --limit 300
  # 指定某天取 500 局
  python tools/download_replays.py --date 2026-06-25 --limit 500
  # 整天全下（注意每天约 20GB！）
  python tools/download_replays.py --date 2026-06-25 --limit 0
  # 最近 3 天各自全下
  python tools/download_replays.py --latest 3 --limit 0
"""
import argparse
import sys
import time
from pathlib import Path

import pandas as pd
from kaggle.api.kaggle_api_extended import KaggleApi
from requests.exceptions import HTTPError

MANIFEST = Path("data/episodes-index/manifest.csv")
OUT_DEFAULT = Path("data/replay")


def pick_dates(args, manifest: pd.DataFrame) -> list[tuple[str, str]]:
    """返回 [(date, daily_dataset_slug), ...]，按日期升序。"""
    m = manifest.sort_values("date")
    if args.date:
        m = m[m["date"].isin(args.date)]
    elif args.latest:
        m = m.tail(args.latest)
    # --all -> 不过滤
    if m.empty:
        sys.exit(f"manifest 里没匹配到日期；可选：{list(manifest['date'])}")
    return list(zip(m["date"], m["daily_dataset_slug"]))


def _with_retry(fn, *, what: str, retries: int = 5):
    """Kaggle 接口偶发 403 限流 → 指数退避重试。"""
    for attempt in range(1, retries + 1):
        try:
            return fn()
        except HTTPError as e:
            code = getattr(e.response, "status_code", None)
            if code in (403, 429) and attempt < retries:
                wait = 20 * attempt  # 列文件接口限流窗口较长，退避要够久
                print(f"  …{what} 被限流({code})，{wait}s 后重试 [{attempt}/{retries}]")
                time.sleep(wait)
                continue
            raise


def day_file_index(api: KaggleApi, slug: str) -> list[str]:
    """返回某日数据集的全部 <EpisodeId>.json 文件名，带本地缓存。

    Kaggle 的 ListDatasetFiles 接口限流很狠，所以列一次就缓存到
    data/episodes-index/files_<date>.txt，之后直接读缓存、不再调接口。
    """
    date = slug.rsplit("episodes-", 1)[-1]  # ...-episodes-2026-06-25 -> 2026-06-25
    cache = MANIFEST.parent / f"files_{date}.txt"
    if cache.exists():
        return cache.read_text().split()

    names, token = [], None
    while True:
        r = _with_retry(lambda: api.dataset_list_files(slug, page_token=token, page_size=200),
                        what="列文件")
        names.extend(f.name for f in r.files if f.name.endswith(".json"))
        token = getattr(r, "nextPageToken", None)
        if not token:
            break
        time.sleep(2.0)  # 翻页间隔，缓和列文件接口限流
    cache.write_text("\n".join(names))
    print(f"  已缓存 {len(names)} 个文件名 -> {cache}")
    return names


def main():
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group()
    g.add_argument("--date", nargs="*", help="指定日期 YYYY-MM-DD（可多个）")
    g.add_argument("--latest", type=int, help="manifest 里最近 N 天")
    g.add_argument("--all", action="store_true", help="manifest 里全部日期")
    ap.add_argument("--limit", type=int, default=300,
                    help="每天最多取多少局；0 = 整天全下（约 20GB/天）")
    ap.add_argument("--out", type=Path, default=OUT_DEFAULT)
    ap.add_argument("--sleep", type=float, default=0.3, help="每个文件下载后的间隔秒，降低限流风险")
    args = ap.parse_args()

    if not MANIFEST.exists():
        sys.exit(f"找不到 {MANIFEST}；先跑：kaggle datasets download "
                 "kaggle/pokemon-tcg-ai-battle-episodes-index -p data/episodes-index --unzip")
    manifest = pd.read_csv(MANIFEST)
    targets = pick_dates(args, manifest)

    api = KaggleApi()
    api.authenticate()  # 读 KAGGLE_API_TOKEN / kaggle.json

    args.out.mkdir(parents=True, exist_ok=True)
    existing = {p.name for p in args.out.glob("*.json")}
    grand_ok = grand_fail = 0

    for date, slug in targets:
        print(f"\n=== {date}  ({slug}) ===")
        if args.limit == 0:
            # 整天全下：一次性 zip 下载再解压（最快）
            print("  整天 bulk 下载（约 20GB，耐心等）…")
            api.dataset_download_files(slug, path=str(args.out), unzip=True, quiet=False)
            grand_ok += 1
            continue

        all_names = day_file_index(api, slug)
        todo = [n for n in all_names if n not in existing][:args.limit]
        print(f"  当天共 {len(all_names)} 局，本次取 {len(todo)} 局（已跳过本地已存在的）")
        for i, name in enumerate(todo, 1):
            try:
                _with_retry(lambda: api.dataset_download_file(slug, name, path=str(args.out), quiet=True),
                            what=f"下载 {name}")
                existing.add(name)
                grand_ok += 1
            except Exception as e:  # noqa: BLE001 兜底，不让单局失败中断整批
                print(f"  ! {name} 失败：{e}")
                grand_fail += 1
            if i % 50 == 0:
                print(f"  进度 {i}/{len(todo)}  ok={grand_ok} fail={grand_fail}")
            time.sleep(args.sleep)  # 温和间隔，降低触发限流概率

    print(f"\n完成：新下载 {grand_ok}，失败 {grand_fail}。落地目录 {args.out}/")


if __name__ == "__main__":
    main()
