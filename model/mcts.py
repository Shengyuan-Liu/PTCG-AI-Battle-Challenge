"""MCTS（蒙特卡洛树搜索）+ 改进的隐藏信息 determinization。

树搜索骨架忠实移植自官方样例（Node/Child/create_node/PUCT/选最多访问）。
**唯一实质改动**：根节点 search_begin 的 determinization——把样例「对手全填 Snorlax /
基础能量」换成 prior.OpponentPrior 的数据驱动采样（见 determinize）。无 prior 时退回常量。
"""
import math
import random

from cg.api import CardType, SearchState, search_begin, search_end, search_step, to_observation_class

from .encoding import card_table, enumerate_actions, get_decoder_input, get_encoder_input
from .network import eval_nn
from .prior import OpponentPrior

SEARCH_COUNT = 10          # 单步 MCTS 次数（受 10min/CPU 预算限制，样例默认 10）
SNORLAX, BASIC_ENERGY = 1072, 1   # 无 prior 时的常量填充（同样例）


def _basic_pokemon(pool, rng) -> int:
    """从候选里挑一张基础宝可梦做对手暗置 active（search_begin 要求 active 是宝可梦）。"""
    cands = [cid for cid in pool
             if (cd := card_table.get(cid)) and cd.cardType == CardType.POKEMON and cd.basic]
    return rng.choice(cands) if cands else SNORLAX


class LearnSample:
    def __init__(self, value, policy, sv_enc, sv_dec):
        self.value = value
        self.policy = policy
        self.sv_enc = sv_enc
        self.sv_dec = sv_dec


class Child:
    def __init__(self, select: list[int], prob: float):
        self.node = None
        self.select = select
        self.prob = prob


class Node:
    def __init__(self, parent, state: SearchState):
        self.value = -2.0
        self.total = 0.0
        self.visit = 0
        self.parent = parent
        self.children: list[Child] = []
        self.state = state

    def backprop(self, value: float):
        self.total += value
        self.visit += 1
        if self.parent is not None:
            self.parent.backprop(value)


def determinize(obs, your_deck: list[int], opp_prior: OpponentPrior, rng: random.Random) -> dict:
    """构造 search_begin 的隐藏信息预测。改进点：对手未知卡用真实卡组先验采样。"""
    state = obs.current
    yi = state.yourIndex
    me = state.players[yi]
    opp = state.players[1 - yi]

    # 我方（已知牌表里采样剩余牌库/奖赏；与样例一致）
    k_deck = min(me.deckCount, len(your_deck))
    your_deck_s = rng.sample(your_deck, k_deck)
    your_prize_s = rng.sample(your_deck, min(len(me.prize), len(your_deck)))

    # 对手暗置 active 是否需要预测
    opp_active_hidden = len(opp.active) > 0 and opp.active[0] is None
    need = opp.deckCount + opp.handCount + len(opp.prize) + (1 if opp_active_hidden else 0)

    if opp_prior.available:
        pool = opp_prior.sample_cards(need, rng)              # 改进：真实卡
    else:
        pool = [SNORLAX] * need                               # fallback：常量
    if len(pool) < need:                                     # 兜底补齐，保证计数足够
        pool += [SNORLAX] * (need - len(pool))

    i = 0
    opp_deck = pool[i:i + opp.deckCount]; i += opp.deckCount
    opp_hand = pool[i:i + opp.handCount]; i += opp.handCount
    opp_prize = pool[i:i + len(opp.prize)]; i += len(opp.prize)
    # 暗置 active 必须是基础宝可梦
    opp_active = [_basic_pokemon(pool or opp_deck, rng)] if opp_active_hidden else []

    if not opp_prior.available:                  # fallback：同样例的常量填充
        opp_deck = [SNORLAX] * opp.deckCount
        opp_hand = [BASIC_ENERGY] * opp.handCount
        opp_prize = [BASIC_ENERGY] * len(opp.prize)
    return dict(
        your_deck=your_deck_s, your_prize=your_prize_s,
        opponent_deck=opp_deck, opponent_prize=opp_prize,
        opponent_hand=opp_hand, opponent_active=opp_active,
    )


def create_node(parent, search_state: SearchState, your_index: int,
                your_deck: list[int], model):
    node = Node(parent, search_state)
    obs = search_state.observation
    state = obs.current
    if state.result >= 0:
        node.value = 0 if state.result == 2 else (1 if state.result == your_index else -1)
        node.backprop(node.value)
        return node, None

    actions = enumerate_actions(obs)
    sv_enc = get_encoder_input(obs, your_deck)
    sv_dec = get_decoder_input(obs, actions)
    value, policy = eval_nn(sv_enc, sv_dec, model)
    v = value if state.yourIndex == your_index else -value
    node.value = v
    node.backprop(v)

    total = 0.0
    for i in range(len(policy)):
        p = math.exp(policy[i] * 10.0)
        node.children.append(Child(actions[i], p))
        total += p
    for c in node.children:
        c.prob /= total
    return node, LearnSample(value, policy, sv_enc, sv_dec)


def mcts_agent(obs_dict: dict, your_deck: list[int], model,
               opp_prior: OpponentPrior, num_searches: int = SEARCH_COUNT,
               rng: random.Random = random):
    """跑一次 MCTS，返回 (选择的 option 下标列表, 训练样本 LearnSample)。"""
    obs = to_observation_class(obs_dict)
    your_index = obs.current.yourIndex

    det = determinize(obs, your_deck, opp_prior, rng)
    search_state = search_begin(obs, **det)
    root, sample = create_node(None, search_state, your_index, your_deck, model)

    for _ in range(num_searches):
        current = root
        while True:
            best, best_v = None, -1e9
            c = 0.4 * math.sqrt(current.visit)
            for child in current.children:
                visit = 0
                if child.node is None:
                    v = current.total / current.visit
                else:
                    v = child.node.total / child.node.visit
                    visit = child.node.visit
                if current.state.observation.current.yourIndex != your_index:
                    v = -v
                v += c * child.prob / (1 + visit)
                if best_v < v:
                    best_v, best = v, child
            if best is None:
                break
            if best.node is None:
                search_state = search_step(current.state.searchId, best.select)
                best.node, _ = create_node(current, search_state, your_index, your_deck, model)
                break
            current = best.node
            if current.state.observation.current.result >= 0:
                current.backprop(current.value)
                break

    # 选访问次数最多的子节点出手
    max_child, max_visit, min_value = None, -1, 10.0
    for child in root.children:
        if child.node is not None:
            if max_visit < child.node.visit:
                max_child, max_visit = child, child.node.visit
            v = child.node.total / child.node.visit
            min_value = min(min_value, v)

    # 生成训练标签（同样例：value=根均值；policy=子优势裁剪）
    if sample is not None and root.children:
        sample.value = root.total / root.visit
        for i, child in enumerate(root.children):
            if i >= len(sample.policy):
                break
            if child.node is None:
                v = min_value - sample.value - 0.03
            else:
                v = child.node.total / child.node.visit - sample.value
            sample.policy[i] = max(-1.0, min(1.0, v))

    search_end()
    if max_child is None:                       # 极端兜底：没展开任何子节点
        n = len(obs.select.option)
        return (random.sample(range(n), min(obs.select.maxCount, n)) if n else []), sample
    return max_child.select, sample
