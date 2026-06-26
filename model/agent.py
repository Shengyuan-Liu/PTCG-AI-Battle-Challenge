"""提交契约封装：agent(obs_dict) -> list[int]。

加载训练好的权重 + MCTS 出招；任何异常/缺失都回退到合法默认或随机合法选择，
绝不崩溃、绝不返回非法动作（CLAUDE.md §5 的底线）。

接入提交：在 sample_submission/main.py 里
    from model.agent import agent
即可（需把 model/ 一起打进 submission，且权重文件随包）。
"""
import csv
import os
import random

_MODEL = None
_DECK = None
_PRIOR = None
_DECK_PATH = os.path.join(os.path.dirname(__file__), "..", "sample_submission", "deck.csv")
_WEIGHTS = os.path.join(os.path.dirname(__file__), "out", "model_final.pth")
SEARCHES = 10


def _load_deck():
    global _DECK
    if _DECK is None:
        try:
            with open(_DECK_PATH, newline="") as f:
                _DECK = [int(r[0]) for r in csv.reader(f) if r and r[0].strip().isdigit()]
        except Exception:
            _DECK = []
    return _DECK


def _load_model():
    """懒加载模型；失败返回 None（agent 自动降级为随机合法）。"""
    global _MODEL, _PRIOR
    if _MODEL is not None:
        return _MODEL
    try:
        import torch
        from .network import new_model
        from .prior import OpponentPrior, load_prior
        m = new_model(torch.device("cpu"))
        for path in (_WEIGHTS, os.path.join(os.path.dirname(__file__), "out", "sl_init.pth")):
            if os.path.exists(path):
                m.load_state_dict(torch.load(path, map_location="cpu"))
                break
        m.eval()
        _MODEL = m
        _PRIOR = OpponentPrior(load_prior())
    except Exception:
        _MODEL = None
    return _MODEL


def _sanitize(select, obs_dict) -> list[int]:
    """保证返回合法：长度 ∈ [minCount,maxCount]、元素互异且在范围内。"""
    try:
        sd = obs_dict["select"]
        n = len(sd["option"])
        lo, hi = sd.get("minCount", 0), sd.get("maxCount", 0)
    except Exception:
        return list(select) if isinstance(select, list) else []
    seen, out = set(), []
    for i in (select or []):
        if isinstance(i, int) and 0 <= i < n and i not in seen:
            seen.add(i); out.append(i)
        if len(out) >= hi:
            break
    for i in range(n):                       # 不足 minCount 时补齐
        if len(out) >= lo:
            break
        if i not in seen:
            seen.add(i); out.append(i)
    return out


def agent(obs_dict: dict) -> list[int]:
    # 初始选卡组阶段：返回 60 张 card ID（不是下标）
    if obs_dict.get("select") is None and obs_dict.get("current") is None:
        deck = _load_deck()
        if len(deck) == 60:
            return deck
    try:
        model = _load_model()
        if model is not None and obs_dict.get("current") is not None:
            from .mcts import mcts_agent
            sel, _ = mcts_agent(obs_dict, _load_deck(), model, _PRIOR, num_searches=SEARCHES)
            return _sanitize(sel, obs_dict)
    except Exception:
        pass
    # 兜底：随机合法
    try:
        sd = obs_dict["select"]
        n = len(sd["option"])
        return _sanitize(random.sample(range(n), min(sd.get("maxCount", 0), n)) if n else [], obs_dict)
    except Exception:
        return []
