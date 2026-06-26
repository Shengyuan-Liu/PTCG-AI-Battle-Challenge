"""并行竞技场：让两个 agent 头对头多局对战，给出判别性胜率。

打随机对手区分不出强弱（都 90%+）。这里支持更强的固定对手：
  agent 规格字符串：
    "model:model/out/xxx.pth"  —— 用该权重 + MCTS
    "rule"                     —— sample_submission/main.py 的 rule-based agent
    "random"                   —— 随机合法
用法：
  python -m model.arena --a model:model/out/sl_init.pth --b rule --games 100 --workers 24
  python -m model.arena --a model:model/out/model4.pth --b model:model/out/sl_init.pth --games 120
"""
import argparse
import multiprocessing as mp
import os
import random

import torch

from .mcts import mcts_agent
from .network import new_model
from .prior import OpponentPrior, load_prior
from .selfplay import load_deck

_S = {}


def _pool_init(deck, prior_dict, searches):
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    torch.set_num_threads(1)
    _S["deck"] = deck
    _S["prior"] = OpponentPrior(prior_dict)
    _S["searches"] = searches
    _S["models"] = {}
    _S["rule"] = None


def _get_model(path):
    if path not in _S["models"]:
        m = new_model(torch.device("cpu"))
        if os.path.exists(path):
            m.load_state_dict(torch.load(path, map_location="cpu"))
        m.eval()
        _S["models"][path] = m
    return _S["models"][path]


def _act(spec, obs):
    from cg.api import to_observation_class
    if spec.startswith("model:"):
        model = _get_model(spec[6:])
        sel, _ = mcts_agent(obs, _S["deck"], model, _S["prior"], num_searches=_S["searches"])
        return sel
    if spec == "rule":
        if _S["rule"] is None:
            from sample_submission.main import agent as rule_agent
            _S["rule"] = rule_agent
        return _S["rule"](obs)
    o = to_observation_class(obs); n = len(o.select.option)
    return random.sample(range(n), min(o.select.maxCount, n)) if n else []


def _game(args):
    """A=spec_a, B=spec_b；seed 决定谁先手。返回 +1=A胜 / -1=B胜 / 0=平。"""
    spec_a, spec_b, seed = args
    from cg.game import battle_finish, battle_select, battle_start
    deck = _S["deck"]
    a_index = seed % 2                       # A 当哪个玩家（轮流先后手）
    with torch.inference_mode():
        obs, _ = battle_start(deck, deck)
        steps = 0
        while obs["current"]["result"] < 0 and steps < 5000:
            spec = spec_a if obs["current"]["yourIndex"] == a_index else spec_b
            obs = battle_select(_act(spec, obs))
            steps += 1
        result = obs["current"]["result"]
        battle_finish()
    if result == 2:
        return 0
    return 1 if result == a_index else -1


def run(spec_a, spec_b, games, workers, searches):
    deck = load_deck()
    prior = load_prior()
    ctx = mp.get_context("spawn")
    with ctx.Pool(workers, initializer=_pool_init, initargs=(deck, prior, searches)) as pool:
        res = pool.map(_game, [(spec_a, spec_b, s) for s in range(games)])
    a_win = sum(x == 1 for x in res); b_win = sum(x == -1 for x in res); draw = sum(x == 0 for x in res)
    decisive = a_win + b_win
    wr = 100 * a_win / max(decisive, 1)
    print(f"[{spec_a}]  vs  [{spec_b}]   games={games}")
    print(f"  A wins={a_win}  B wins={b_win}  draw={draw}  ->  A win% (decisive) = {wr:.0f}%")
    return wr


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", required=True)
    ap.add_argument("--b", required=True)
    ap.add_argument("--games", type=int, default=100)
    ap.add_argument("--workers", type=int, default=24)
    ap.add_argument("--searches", type=int, default=10)
    a = ap.parse_args()
    run(a.a, a.b, a.games, a.workers, a.searches)
