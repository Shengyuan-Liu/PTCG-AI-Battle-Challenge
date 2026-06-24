"""PTCG AI Battle Challenge — rule-based agent (v1).

设计目标（优先级从高到低）：
1. **绝不崩溃、绝不超时、永远返回合法动作**——任何异常都回退到合法默认值。
   合法性约束：返回 option 下标列表，长度 ∈ [minCount, maxCount]，元素互不相同且 0<=i<len(option)。
2. 在高价值、低风险的决策点用启发式（布场、进化、贴能量、过牌、攻击择优、先后手、mulligan、数量取大）。
3. 其余未专门处理的选择 → 合法默认（取前 minCount 个），由 `_sanitize` 统一保证合法。

注意：本文件在 Kaggle(Linux x86-64) 上运行；`import cg.api` 会加载原生引擎库。
"""

import os

from cg.api import (
    to_observation_class,
    SelectType, SelectContext, OptionType, AreaType, CardType,
    all_card_data, all_attack,
)

# ----------------------------------------------------------------------------
# 静态数据缓存（懒加载；失败则为空 dict，启发式自动降级为合法默认，不影响合法性）
# ----------------------------------------------------------------------------
_CARD = None   # cardId -> CardData
_ATK = None    # attackId -> Attack


def _card_db():
    global _CARD
    if _CARD is None:
        db = {}
        try:
            for c in all_card_data():
                db[c.cardId] = c
        except Exception:
            db = {}
        _CARD = db
    return _CARD


def _atk_db():
    global _ATK
    if _ATK is None:
        db = {}
        try:
            for a in all_attack():
                db[a.attackId] = a
        except Exception:
            db = {}
        _ATK = db
    return _ATK


# ----------------------------------------------------------------------------
# 卡组（初始选卡阶段返回 60 个 card ID）
# ----------------------------------------------------------------------------
def read_deck_csv():
    path = "deck.csv"
    if not os.path.exists(path):
        path = "/kaggle_simulations/agent/deck.csv"
    with open(path, "r") as f:
        ids = [int(x) for x in f.read().split() if x.strip()]
    return ids[:60]


# ----------------------------------------------------------------------------
# 每回合状态（防死循环：限制单回合动作数与特性使用次数）
# ----------------------------------------------------------------------------
_turn = {"t": None, "act": 0, "abil": 0}
_MAX_ACTIONS = 60   # 单回合 MAIN 动作上限，超过强制结束/攻击
_MAX_ABILITY = 10   # 单回合特性使用上限


def _turn_tick(state):
    t = getattr(state, "turn", None)
    if _turn["t"] != t:
        _turn["t"] = t
        _turn["act"] = 0
        _turn["abil"] = 0
    _turn["act"] += 1


# ----------------------------------------------------------------------------
# 小工具
# ----------------------------------------------------------------------------
def _by_type(options, *types):
    want = set(int(t) for t in types)
    return [i for i, o in enumerate(options) if int(o.type) in want]


def _player(state, pidx):
    try:
        return state.players[pidx]
    except Exception:
        return None


def _resolve(obs, opt):
    """把一个 option 指向的卡解析成 state 里的 Card/Pokemon 对象（取不到返回 None）。"""
    try:
        st = obs.current
        if opt.area is None or opt.index is None:
            return None
        area = int(opt.area)
        idx = opt.index
        pidx = opt.playerIndex if opt.playerIndex is not None else (st.yourIndex if st else 0)
        if area == AreaType.HAND:
            p = _player(st, pidx)
            return p.hand[idx] if (p and p.hand) else None
        if area == AreaType.BENCH:
            p = _player(st, pidx)
            return p.bench[idx] if p else None
        if area == AreaType.ACTIVE:
            p = _player(st, pidx)
            return p.active[idx] if p else None
        if area == AreaType.DISCARD:
            p = _player(st, pidx)
            return p.discard[idx] if p else None
        if area == AreaType.PRIZE:
            p = _player(st, pidx)
            return p.prize[idx] if p else None
        if area == AreaType.STADIUM:
            return st.stadium[idx]
        if area == AreaType.DECK:
            return obs.select.deck[idx] if obs.select.deck else None
    except Exception:
        return None
    return None


def _hp_of(obj):
    try:
        if obj is None:
            return -1
        hp = getattr(obj, "hp", None)        # Pokemon 对象自带当前 hp
        if hp is not None:
            return hp
        cd = _card_db().get(getattr(obj, "id", None))
        return cd.hp if (cd and cd.hp) else 0
    except Exception:
        return -1


def _is_basic(card):
    try:
        cd = _card_db().get(card.id)
        return bool(cd and cd.basic)
    except Exception:
        return False


def _opt_pokemon_score(obs, opt):
    """用于晋位/布场：优先已带能量（能立即攻击），其次高 HP。"""
    c = _resolve(obs, opt)
    if c is None:
        return -1
    energies = len(getattr(c, "energies", None) or [])
    return energies * 1000 + max(0, _hp_of(c))


def _opt_value_score(obs, opt):
    """卡价值粗排：宝可梦 > 训练家 > 其它（用于检索拿牌 / 弃牌）。"""
    c = _resolve(obs, opt)
    if c is None:
        return 0
    cd = _card_db().get(getattr(c, "id", None))
    if cd is None:
        return 1
    ct = int(cd.cardType)
    if ct == CardType.POKEMON:
        return 3
    if ct in (CardType.SUPPORTER, CardType.ITEM):
        return 2
    return 1


def _legal(sel):
    """通用合法默认：取前 minCount 个下标。"""
    n = len(sel.option)
    mn = max(0, min(int(sel.minCount), n))
    return list(range(mn))


# ----------------------------------------------------------------------------
# 攻击择优
# ----------------------------------------------------------------------------
def _opp_active_hp(obs):
    try:
        st = obs.current
        opp = _player(st, 1 - st.yourIndex)
        if opp and opp.active and opp.active[0] is not None:
            return opp.active[0].hp
    except Exception:
        pass
    return None


def _best_attack_idx(obs, indices):
    """在给定的攻击 option 下标里选：能击倒优先，其次基础伤害最高。"""
    opts = obs.select.option
    opp_hp = _opp_active_hp(obs)
    adb = _atk_db()

    def score(i):
        aid = opts[i].attackId
        a = adb.get(aid) if aid is not None else None
        dmg = (a.damage if (a and a.damage is not None) else 0)
        ko = 1 if (opp_hp is not None and dmg > 0 and dmg >= opp_hp) else 0
        return (ko, dmg)

    return max(indices, key=score)


# ----------------------------------------------------------------------------
# YES / NO
# ----------------------------------------------------------------------------
def _has_basic_in_hand(obs):
    try:
        st = obs.current
        me = _player(st, st.yourIndex)
        if not me or not me.hand:
            return False
        return any(_is_basic(c) for c in me.hand)
    except Exception:
        return False


def _yes_no(obs):
    sel = obs.select
    opts = sel.option
    ctx = int(sel.context)
    yes = next((i for i, o in enumerate(opts) if int(o.type) == OptionType.YES), None)
    no = next((i for i, o in enumerate(opts) if int(o.type) == OptionType.NO), None)

    def pick(want_yes):
        if want_yes and yes is not None:
            return yes
        if (not want_yes) and no is not None:
            return no
        return yes if yes is not None else (no if no is not None else 0)

    if ctx == SelectContext.IS_FIRST:
        return pick(True)                              # 进化系卡组：先手多一回合铺场
    if ctx == SelectContext.MULLIGAN:
        return pick(not _has_basic_in_hand(obs))       # 没有基础宝可梦才重抽
    if ctx == SelectContext.ACTIVATE:
        return pick(True)                              # 可选效果一般有利，发动
    if ctx == SelectContext.FIRST_EFFECT:
        return pick(True)
    if ctx == SelectContext.MORE_DEVOLVE:
        return pick(False)                             # 保守，不继续退化
    if ctx == SelectContext.COIN_HEAD:
        return pick(True)
    return pick(True)


# ----------------------------------------------------------------------------
# CARD 选择（按 context）
# ----------------------------------------------------------------------------
def _card_select(obs):
    sel = obs.select
    opts = sel.option
    n = len(opts)
    ctx = int(sel.context)

    # 选一只上场 / 晋位 / 交换 → 选最能打的（带能量、高 HP）
    if ctx in (SelectContext.SETUP_ACTIVE_POKEMON, SelectContext.SWITCH, SelectContext.TO_ACTIVE):
        return [max(range(n), key=lambda k: _opt_pokemon_score(obs, opts[k]))]

    # 铺场 → 在允许范围内尽量多放（优先高 HP），增加板面
    if ctx in (SelectContext.SETUP_BENCH_POKEMON, SelectContext.TO_BENCH, SelectContext.TO_FIELD):
        order = sorted(range(n), key=lambda k: _opt_pokemon_score(obs, opts[k]), reverse=True)
        k = min(int(sel.maxCount), n)
        k = max(k, int(sel.minCount))
        return order[:k]

    # 检索拿牌 → 优先宝可梦/关键卡
    if ctx == SelectContext.TO_HAND:
        order = sorted(range(n), key=lambda k: _opt_value_score(obs, opts[k]), reverse=True)
        k = int(sel.minCount) if int(sel.minCount) > 0 else min(1, n)
        k = min(k, int(sel.maxCount))
        return order[:k]

    # 弃牌 / 回库 → 优先丢价值最低的
    if ctx in (SelectContext.DISCARD, SelectContext.TO_DECK, SelectContext.TO_DECK_BOTTOM):
        order = sorted(range(n), key=lambda k: _opt_value_score(obs, opts[k]))
        k = int(sel.minCount)
        return order[:k]

    return _legal(sel)


# ----------------------------------------------------------------------------
# MAIN（每次调用选一个动作；多次调用串成一整个回合）
# ----------------------------------------------------------------------------
def _best_attach(obs, attaches):
    opts = obs.select.option
    for i in attaches:                                  # 优先贴到出战位（主攻手）
        o = opts[i]
        if o.inPlayArea is not None and int(o.inPlayArea) == AreaType.ACTIVE:
            return i
    return attaches[0]


def _best_play(obs, plays):
    opts = obs.select.option

    def rank(i):
        c = _resolve(obs, opts[i])
        cd = _card_db().get(getattr(c, "id", None)) if c else None
        if cd is None:
            return 1
        ct = int(cd.cardType)
        if ct == CardType.SUPPORTER:
            return 4                                     # 过牌/检索，价值最高
        if ct == CardType.POKEMON and cd.basic:
            return 3                                     # 铺基础宝可梦
        if ct == CardType.ITEM:
            return 2
        return 1

    return max(plays, key=rank)


def _main(obs):
    sel = obs.select
    opts = sel.option
    _turn_tick(obs.current)

    ends = _by_type(opts, OptionType.END)
    attacks = _by_type(opts, OptionType.ATTACK)
    evolves = _by_type(opts, OptionType.EVOLVE)
    attaches = _by_type(opts, OptionType.ATTACH)
    plays = _by_type(opts, OptionType.PLAY)
    abilities = _by_type(opts, OptionType.ABILITY)

    # 安全阀：回合动作过多 → 强制攻击或结束，避免死循环/超时
    if _turn["act"] > _MAX_ACTIONS:
        if attacks:
            return [_best_attack_idx(obs, attacks)]
        if ends:
            return [ends[0]]
        return _legal(sel)

    # 标准出牌顺序：先做不结束回合的准备动作，最后攻击
    if evolves:
        return [evolves[0]]
    if attaches:
        return [_best_attach(obs, attaches)]
    if plays:
        return [_best_play(obs, plays)]
    if abilities and _turn["abil"] < _MAX_ABILITY:
        _turn["abil"] += 1
        return [abilities[0]]
    if attacks:
        return [_best_attack_idx(obs, attacks)]
    if ends:
        return [ends[0]]
    return _legal(sel)


# ----------------------------------------------------------------------------
# 分派
# ----------------------------------------------------------------------------
def _choose(obs):
    sel = obs.select
    stype = int(sel.type)

    if stype == SelectType.MAIN:
        return _main(obs)
    if stype == SelectType.YES_NO:
        return [_yes_no(obs)]
    if stype == SelectType.COUNT:
        opts = sel.option
        return [max(range(len(opts)),
                    key=lambda i: (opts[i].number if opts[i].number is not None else -1))]
    if stype == SelectType.ATTACK:
        return [_best_attack_idx(obs, list(range(len(sel.option))))]
    if stype == SelectType.CARD:
        return _card_select(obs)
    # ENERGY / EVOLVE / SKILL / ATTACHED_CARD / SPECIAL_CONDITION / ... → 合法默认
    return _legal(sel)


# ----------------------------------------------------------------------------
# 合法化兜底：无论 _choose 返回什么，这里都保证输出合法
# ----------------------------------------------------------------------------
def _sanitize(choice, sel, obs_dict):
    try:
        n = len(sel.option)
        mn = int(sel.minCount)
        mx = int(sel.maxCount)
    except Exception:
        return _raw_fallback(obs_dict)
    mn = max(0, min(mn, n))
    mx = max(mn, min(mx, n))

    out = []
    if choice:
        for i in choice:
            try:
                i = int(i)
            except Exception:
                continue
            if 0 <= i < n and i not in out:
                out.append(i)
                if len(out) >= mx:
                    break
    if len(out) < mn:                                   # 补足到 minCount
        for i in range(n):
            if i not in out:
                out.append(i)
                if len(out) >= mn:
                    break
    return out[:mx]


def _raw_fallback(obs_dict):
    """连 dataclass 都转换失败时，直接用原始 dict 给出合法返回。"""
    try:
        sel = obs_dict.get("select")
        if sel is None:
            return read_deck_csv()
        n = len(sel["option"])
        mn = max(0, min(int(sel.get("minCount", 0)), n))
        return list(range(mn))
    except Exception:
        return []


# ----------------------------------------------------------------------------
# 入口
# ----------------------------------------------------------------------------
def agent(obs_dict):
    """Return option index list (or 60 card IDs at initial deck selection)."""
    try:
        obs = to_observation_class(obs_dict)
    except Exception:
        return _raw_fallback(obs_dict)

    if obs.select is None:
        try:
            return read_deck_csv()
        except Exception:
            return _raw_fallback(obs_dict)

    try:
        choice = _choose(obs)
    except Exception:
        choice = None
    return _sanitize(choice, obs.select, obs_dict)
