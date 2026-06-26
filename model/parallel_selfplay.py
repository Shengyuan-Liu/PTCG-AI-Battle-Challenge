"""最优本地训练：多进程并行 self-play（CPU 多核喂数据）+ 主进程 GPU 训练。

为什么这样最优（见 docs/model.md §7 与对话记录）：
  - 引擎是**单例**，一个进程只能串行跑一局 → 用 N 个进程各自独立加载引擎并行自对弈。
  - MCTS 是「单局面」小前向，GPU 每次调用开销大 → worker 用 **CPU 推理**（且每进程
    限 1 线程，避免 32 进程互相抢核），把 CPU 多核打满。
  - **GPU 只用于训练步**（batch 梯度更新），那里 GPU 真正快。

流程：每轮 → 存当前权重 → N worker 并行(eval + 自对弈) → 主进程 GPU 训练 → 存档 → 下一轮。
"""
import argparse
import multiprocessing as mp
import os
import random
import time

import torch

from .mcts import mcts_agent
from .network import new_model
from .prior import OpponentPrior, load_prior
from .selfplay import load_deck, train_on

# ---- worker 端全局状态（每个子进程一份）-----------------------------------
_S = {"version": None, "model": None, "deck": None, "prior": None, "searches": 10}


def _pool_init(deck, prior_dict, searches):
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    torch.set_num_threads(1)                       # 关键：每 worker 单线程，避免抢核
    _S["deck"] = deck
    _S["prior"] = OpponentPrior(prior_dict)
    _S["searches"] = searches


def _ensure_model(weights_path, version):
    if _S["version"] != version:
        m = new_model(torch.device("cpu"))
        if weights_path and os.path.exists(weights_path):
            m.load_state_dict(torch.load(weights_path, map_location="cpu"))
        m.eval()
        _S["model"], _S["version"] = m, version


def _selfplay_task(args):
    weights_path, version, seed = args
    _ensure_model(weights_path, version)
    from cg.game import battle_finish, battle_select, battle_start
    deck, prior, model, searches = _S["deck"], _S["prior"], _S["model"], _S["searches"]
    rng = random.Random(seed)
    samples = [[], []]
    with torch.inference_mode():
        obs, _ = battle_start(deck, deck)
        steps = 0
        while obs["current"]["result"] < 0 and steps < 5000:
            sel, s = mcts_agent(obs, deck, model, prior, num_searches=searches, rng=rng)
            if s is not None:
                samples[obs["current"]["yourIndex"]].append(s)
            obs = battle_select(sel)
            steps += 1
        result = obs["current"]["result"]
        battle_finish()
    out = []
    for i in range(2):                              # TD(λ) value 标签
        value = 1.0 if i == result else -1.0
        for s in reversed(samples[i]):
            s.value, value = (value + s.value) * 0.5, value * 0.9 + s.value * 0.1
            out.append(s)
    return out


def _eval_task(args):
    weights_path, version, seed = args
    _ensure_model(weights_path, version)
    from cg.api import to_observation_class
    from cg.game import battle_finish, battle_select, battle_start
    deck, prior, model, searches = _S["deck"], _S["prior"], _S["model"], _S["searches"]
    rng = random.Random(seed)
    yi = seed % 2
    with torch.inference_mode():
        obs, _ = battle_start(deck, deck)
        steps = 0
        while obs["current"]["result"] < 0 and steps < 5000:
            if obs["current"]["yourIndex"] == yi:
                sel, _ = mcts_agent(obs, deck, model, prior, num_searches=searches, rng=rng)
            else:
                o = to_observation_class(obs); n = len(o.select.option)
                sel = rng.sample(range(n), min(o.select.maxCount, n)) if n else []
            obs = battle_select(sel)
            steps += 1
        result = obs["current"]["result"]
        battle_finish()
    return 1 if result == yi else (0 if result == 2 else -1)


def train(iterations=10, workers=16, selfplay_games=64, eval_games=32, searches=10,
          deck_path="sample_submission/deck.csv", init_weights=None, out_dir="model/out"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = new_model(device)
    if init_weights and os.path.exists(init_weights):
        model.load_state_dict(torch.load(init_weights, map_location=device))
        print(f"loaded init weights: {init_weights}")
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
    deck = load_deck(deck_path)
    prior_dict = load_prior()
    os.makedirs(out_dir, exist_ok=True)
    weights_path = os.path.join(out_dir, "_current.pth")
    print(f"device={device} workers={workers} deck={len(deck)} "
          f"prior={'data' if prior_dict.get('decklists') else 'constant'} "
          f"selfplay/iter={selfplay_games} searches={searches}", flush=True)

    ctx = mp.get_context("spawn")
    with ctx.Pool(workers, initializer=_pool_init, initargs=(deck, prior_dict, searches)) as pool:
        for it in range(iterations):
            torch.save(model.state_dict(), weights_path)
            ver = it
            t0 = time.time()
            ev = pool.map(_eval_task, [(weights_path, ver, s) for s in range(eval_games)])
            win = sum(x == 1 for x in ev); lose = sum(x == -1 for x in ev)
            wr = 100 * win / max(win + lose, 1)
            torch.save(model.state_dict(), os.path.join(out_dir, f"model{it}.pth"))

            res = pool.map(_selfplay_task, [(weights_path, ver, 100000 + it * 10000 + s)
                                            for s in range(selfplay_games)])
            samples = [s for game in res for s in game]
            tcollect = time.time() - t0
            t1 = time.time()
            train_on(model, optimizer, samples, device)
            print(f"[iter {it}] win%={wr:.0f}  samples={len(samples)}  "
                  f"collect={tcollect:.0f}s train={time.time()-t1:.1f}s", flush=True)

        torch.save(model.state_dict(), weights_path)
        ev = pool.map(_eval_task, [(weights_path, iterations, s) for s in range(eval_games)])
        win = sum(x == 1 for x in ev); lose = sum(x == -1 for x in ev)
        print(f"[final] win%={100*win/max(win+lose,1):.0f}", flush=True)
    torch.save(model.state_dict(), os.path.join(out_dir, "model_final.pth"))
    print(f"saved → {out_dir}/model_final.pth", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--iterations", type=int, default=10)
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--selfplay-games", type=int, default=64)
    ap.add_argument("--eval-games", type=int, default=32)
    ap.add_argument("--searches", type=int, default=10)
    ap.add_argument("--deck", default="sample_submission/deck.csv")
    ap.add_argument("--init-weights", default=None)
    ap.add_argument("--out-dir", default="model/out")
    a = ap.parse_args()
    train(a.iterations, a.workers, a.selfplay_games, a.eval_games, a.searches,
          a.deck, a.init_weights, a.out_dir)
