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
from .selfplay import load_deck, to_onehot_policy, train_on, train_on_az

# ---- worker 端全局状态（每个子进程一份）-----------------------------------
_S = {"version": None, "model": None, "deck": None, "prior": None, "searches": 10}


def _pool_init(deck, prior_dict, searches, ptarget="advantage"):
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    torch.set_num_threads(1)                       # 关键：每 worker 单线程，避免抢核
    _S["deck"] = deck
    _S["prior"] = OpponentPrior(prior_dict)
    _S["searches"] = searches
    _S["ptarget"] = ptarget


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
    ptarget = _S.get("ptarget", "advantage")
    rng = random.Random(seed)
    samples = [[], []]
    with torch.inference_mode():
        obs, _ = battle_start(deck, deck)
        steps = 0
        while obs["current"]["result"] < 0 and steps < 5000:
            sel, s = mcts_agent(obs, deck, model, prior, num_searches=searches,
                                rng=rng, policy_target=ptarget)
            if s is not None:
                samples[obs["current"]["yourIndex"]].append(s)
            obs = battle_select(sel)
            steps += 1
        result = obs["current"]["result"]
        battle_finish()
    out = []
    for i in range(2):
        if ptarget == "visit":                      # AlphaZero：value 目标 = 最终胜负 z
            z = 1.0 if i == result else (-1.0 if result != 2 else 0.0)
            for s in samples[i]:
                s.value = z
                out.append(s)
        else:                                       # 样例：TD(λ) value 标签
            value = 1.0 if i == result else -1.0
            for s in reversed(samples[i]):
                s.value, value = (value + s.value) * 0.5, value * 0.9 + s.value * 0.1
                out.append(s)
    return out


def _opponent_move(opponent, obs, rng):
    """评估对手的一步：'rule' = 项目 rule-based agent；否则随机合法。"""
    if opponent == "rule":
        if _S.get("rule") is None:
            from sample_submission.main import agent as rule_agent
            _S["rule"] = rule_agent
        return _S["rule"](obs)
    from cg.api import to_observation_class
    o = to_observation_class(obs); n = len(o.select.option)
    return rng.sample(range(n), min(o.select.maxCount, n)) if n else []


def _eval_task(args):
    weights_path, version, seed, opponent = args
    _ensure_model(weights_path, version)
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
                sel = _opponent_move(opponent, obs, rng)
            obs = battle_select(sel)
            steps += 1
        result = obs["current"]["result"]
        battle_finish()
    return 1 if result == yi else (0 if result == 2 else -1)


def train(iterations=10, workers=16, selfplay_games=64, eval_games=32, searches=10,
          deck_path="sample_submission/deck.csv", init_weights=None, out_dir="model/out",
          lr=3e-5, eval_opponent="rule", patience=4, mode="advantage", sl_anchor=True):
    """评估打 rule、自动保留最佳(早停)、低 lr。

    mode="az"：标准 AlphaZero（访问次数分布 policy 标签 + 交叉熵 + value=z），
    并把 replay 行为克隆样本作为 **SL 锚** 混入每轮训练防遗忘（sl_anchor）。
    mode="advantage"：样例原版（会自毁，仅留作对照）。
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = new_model(device)
    if init_weights and os.path.exists(init_weights):
        model.load_state_dict(torch.load(init_weights, map_location=device))
        print(f"loaded init weights: {init_weights}")
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    deck = load_deck(deck_path)
    prior_dict = load_prior()
    os.makedirs(out_dir, exist_ok=True)
    weights_path = os.path.join(out_dir, "_current.pth")
    best_path = os.path.join(out_dir, "model_final.pth")   # 始终指向「目前最好」
    ptarget = "visit" if mode == "az" else "advantage"

    anchor = []
    if mode == "az" and sl_anchor:
        from .imitation import build_dataset
        anchor = [to_onehot_policy(s) for s in build_dataset() if s.value > 0]   # 赢家 one-hot
        print(f"SL anchor samples: {len(anchor)}", flush=True)

    print(f"device={device} workers={workers} deck={len(deck)} mode={mode} "
          f"prior={'data' if prior_dict.get('decklists') else 'constant'} "
          f"selfplay/iter={selfplay_games} searches={searches} lr={lr} eval_vs={eval_opponent}", flush=True)

    best_wr, best_it, stale = -1.0, -1, 0
    ctx = mp.get_context("spawn")
    with ctx.Pool(workers, initializer=_pool_init, initargs=(deck, prior_dict, searches, ptarget)) as pool:
        for it in range(iterations):
            torch.save(model.state_dict(), weights_path)
            ver = it
            t0 = time.time()
            ev = pool.map(_eval_task, [(weights_path, ver, s, eval_opponent) for s in range(eval_games)])
            win = sum(x == 1 for x in ev); lose = sum(x == -1 for x in ev)
            wr = 100 * win / max(win + lose, 1)

            if wr > best_wr:                       # 保留最佳 → model_final.pth（早停核心）
                best_wr, best_it, stale = wr, it, 0
                torch.save(model.state_dict(), best_path)
                tag = "  <- best, saved model_final"
            else:
                stale += 1
                tag = f"  (best={best_wr:.0f}@it{best_it}, stale={stale})"

            res = pool.map(_selfplay_task, [(weights_path, ver, 100000 + it * 10000 + s)
                                            for s in range(selfplay_games)])
            samples = [s for game in res for s in game]
            tcollect = time.time() - t0
            t1 = time.time()
            if mode == "az":
                train_on_az(model, optimizer, samples + anchor, device)   # 混入 SL 锚
            else:
                train_on(model, optimizer, samples, device)
            print(f"[iter {it}] win% vs {eval_opponent}={wr:.0f}  samples={len(samples)}  "
                  f"collect={tcollect:.0f}s train={time.time()-t1:.1f}s{tag}", flush=True)
            if stale >= patience:                  # 连续 patience 轮不再创新高 → 早停
                print(f"early stop: no improvement for {patience} iters", flush=True)
                break

    print(f"[done] best win% vs {eval_opponent} = {best_wr:.0f} @ iter {best_it}  -> {best_path}", flush=True)


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
    ap.add_argument("--lr", type=float, default=3e-5)
    ap.add_argument("--eval-opponent", default="rule", choices=["rule", "random"])
    ap.add_argument("--patience", type=int, default=4)
    ap.add_argument("--mode", default="advantage", choices=["advantage", "az"])
    ap.add_argument("--no-sl-anchor", action="store_true")
    a = ap.parse_args()
    train(a.iterations, a.workers, a.selfplay_games, a.eval_games, a.searches,
          a.deck, a.init_weights, a.out_dir, a.lr, a.eval_opponent, a.patience,
          a.mode, not a.no_sl_anchor)
