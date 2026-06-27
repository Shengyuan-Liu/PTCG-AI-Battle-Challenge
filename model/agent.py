"""提交契约封装：agent(obs_dict) -> list[int]。

加载训练好的权重 + MCTS 出招；任何异常/缺失都回退到合法默认或随机合法选择，
绝不崩溃、绝不返回非法动作（CLAUDE.md §5 的底线）。

接入提交：在 sample_submission/main.py 里
    from model.agent import agent
即可（需把 model/ 一起打进 submission，且权重文件随包）。
"""
import csv
import json
import os
import random

_MODEL = None
_DECK = None
_PRIOR = None
SEARCHES = 10
USE_MCTS = False   # 提交版默认 False：纯 NN 单次前向，避开 search_begin 的原生崩溃风险

# 多候选根目录，兼容本地与 Kaggle(/kaggle_simulations/agent/) 的打包结构
try:
    _HERE = os.path.dirname(os.path.abspath(__file__))
except NameError:
    _HERE = "/kaggle_simulations/agent/model"
_ROOTS = [
    _HERE,                                   # model/
    os.path.join(_HERE, "out"),              # model/out/
    os.path.dirname(_HERE),                  # 包父目录（submission 根）
    "/kaggle_simulations/agent",
    os.path.join("/kaggle_simulations/agent", "model", "out"),
    os.getcwd(),
    os.path.join(_HERE, "..", "sample_submission"),
]


def _find(*names):
    for root in _ROOTS:
        for name in names:
            p = os.path.join(root, name)
            if os.path.exists(p):
                return p
    return None


def _load_deck():
    global _DECK
    if _DECK is None:
        _DECK = []
        path = _find("deck.csv")
        if path:
            try:
                with open(path, newline="") as f:
                    _DECK = [int(r[0]) for r in csv.reader(f) if r and r[0].strip().isdigit()]
            except Exception:
                _DECK = []
    return _DECK


def _load_prior():
    """从打包的 replay_prior.json 读对手卡组先验；找不到则空（mcts 退回常量填充）。"""
    from .prior import OpponentPrior
    path = _find("replay_prior.json")
    if path:
        try:
            return OpponentPrior(json.load(open(path, encoding="utf-8")))
        except Exception:
            pass
    return OpponentPrior({})


def _load_model():
    """懒加载模型；失败返回 None（agent 自动降级，main.py 再兜到 rule-based）。"""
    global _MODEL, _PRIOR
    if _MODEL is not None:
        return _MODEL
    try:
        import torch
        torch.set_num_threads(1)                 # 降内存/CPU 占用
        from .network import new_model
        m = new_model(torch.device("cpu"))
        wpath = _find("model_final.pth", "sl_init.pth")
        if wpath is None:
            return None
        m.load_state_dict(torch.load(wpath, map_location="cpu"))
        m.eval()
        _MODEL = m
        if USE_MCTS:
            _PRIOR = _load_prior()
    except Exception:
        _MODEL = None
    return _MODEL


def _policy_move(obs_dict, model):
    """纯 NN 单次前向选招：编码局面+候选动作 → 取 policy 最高的动作。不调 search_begin。"""
    import torch
    from cg.api import to_observation_class
    from .encoding import enumerate_actions, get_decoder_input, get_encoder_input
    from .network import eval_nn
    obs = to_observation_class(obs_dict)
    actions = enumerate_actions(obs)
    if not actions:
        return []
    sv_e = get_encoder_input(obs, _load_deck())
    sv_d = get_decoder_input(obs, actions)
    with torch.inference_mode():
        _, policy = eval_nn(sv_e, sv_d, model)
    k = max(range(len(actions)), key=lambda j: policy[j] if j < len(policy) else -9.0)
    return actions[k]


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
            if USE_MCTS:
                from .mcts import mcts_agent
                sel, _ = mcts_agent(obs_dict, _load_deck(), model, _PRIOR, num_searches=SEARCHES)
            else:
                sel = _policy_move(obs_dict, model)   # 纯 NN，无 search_begin
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
