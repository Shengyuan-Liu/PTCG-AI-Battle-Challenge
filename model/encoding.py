"""Observation → 稀疏特征编码（encoder/decoder 输入）。

忠实移植自官方样例 other_works/reinforcement-learning-and-mcts-sample-code.ipynb，
仅做模块化与少量注释。核心思路：用 torch.nn.EmbeddingBag(mode="sum") 吃稀疏特征，
靠 offset 把每类特征摆进一个大「虚拟词表」里定位（比 dense one-hot 高效）。

不要随意改这里的 add_pos / 偏移布局——它和 network 的词表大小、decoder 特征下标严格对应。
"""
from cg.api import (
    AreaType, Card, Observation, OptionType, PlayerState, Pokemon, SelectContext,
    all_attack, all_card_data,
)

# ---- 静态尺寸（卡池固定，启动时算一次）-------------------------------------
_all_card = all_card_data()
card_table = {c.cardId: c for c in _all_card}
card_count = max(_all_card, key=lambda c: c.cardId).cardId + 1      # Max Card ID + 1
attack_count = max(all_attack(), key=lambda a: a.attackId).attackId + 1

num_words_encoder = 24          # encoder 的「词」数（见 get_encoder_input）
encoder_size = 22000            # encoder 虚拟词表大小（> 实际用到的下标）

decoder_main_feature = 8        # SelectContext.Main 的特征数
decoder_attack_offset = 14      # Attack 特征起始下标
decoder_card_offset = decoder_attack_offset + attack_count
decoder_size = decoder_card_offset + (1 + decoder_main_feature + SelectContext.RECOVER_SPECIAL_CONDITION) * card_count


class SparseVector:
    """torch.nn.EmbeddingBag 的输入构造器：index / value / offset 三元组。"""
    def __init__(self):
        self.index: list[int] = []
        self.value: list[float] = []
        self.offset: list[int] = []
        self.pos = 0

    def add(self, index: int, value):
        value = float(value)
        if value != 0.0:
            self.index.append(self.pos + index)
            self.value.append(value)

    def add_pos(self, pos: int):
        self.pos += pos

    def add_single(self, value):
        value = float(value)
        if value != 0.0:
            self.index.append(self.pos)
            self.value.append(value)
        self.pos += 1

    def word_start(self):
        self.offset.append(len(self.index))


# ---- encoder 特征 ----------------------------------------------------------
def add_card(sv: SparseVector, card):
    if card is not None:
        sv.add(card.id, 1)
    sv.add_pos(card_count)


def add_cards(sv: SparseVector, cards, value: float):
    if cards is not None:
        for card in cards:
            sv.add(card.id, value)
    sv.add_pos(card_count)


def add_pokemon(sv: SparseVector, poke):
    if poke is None:
        sv.add_single(1)
        sv.add_pos(1 + 3 * card_count)
    else:
        sv.add_single(0)
        sv.add_single(poke.hp / 400)
        add_card(sv, poke)
        add_cards(sv, poke.tools, 1.0)
        add_cards(sv, poke.energyCards, 0.5)


def add_player(sv: SparseVector, ps: PlayerState):
    sv.add_single(ps.deckCount / 60)
    sv.add_single(len(ps.discard) / 60)
    sv.add_single(ps.handCount / 8)
    sv.add_single(len(ps.bench) / 5)
    sv.add(len(ps.prize), 1)
    sv.add_pos(7)
    sv.add_single(ps.poisoned)
    sv.add_single(ps.burned)
    sv.add_single(ps.asleep)
    sv.add_single(ps.paralyzed)
    sv.add_single(ps.confused)
    add_cards(sv, ps.discard, 0.25)


def get_encoder_input(obs: Observation, your_deck: list[int]) -> SparseVector:
    your_index = obs.current.yourIndex
    state = obs.current
    sv = SparseVector()

    for i in range(2):                       # 双方各 8 个 bench 槽
        ps = state.players[i ^ your_index]
        for j in range(8):
            sv.word_start()
            pos = sv.pos
            if j < len(ps.bench):
                add_pokemon(sv, ps.bench[j])
            else:
                add_pokemon(sv, None)
            if j != 7:
                sv.pos = pos                 # 同一「词」里复用位置

    for i in range(2):                       # 双方 active
        ps = state.players[i ^ your_index]
        sv.word_start()
        if 0 < len(ps.active):
            add_pokemon(sv, ps.active[0])
        else:
            add_pokemon(sv, None)

    for i in range(2):                       # 双方玩家汇总
        ps = state.players[i ^ your_index]
        sv.word_start()
        add_player(sv, ps)

    sv.word_start()                          # 我方手牌
    add_cards(sv, state.players[your_index].hand, 0.25)

    sv.word_start()                          # 我方牌库（已知）
    for cid in your_deck:
        sv.add(cid, 0.25)
    sv.add_pos(card_count)

    sv.word_start()                          # 场地
    add_cards(sv, state.stadium, 1.0)

    sv.word_start()                          # 全局：回合 / 先手
    sv.add_single(1)
    sv.add_single(state.turn / 10)
    sv.add_single(state.firstPlayer == your_index)
    return sv


def get_card(obs: Observation, area: AreaType, index: int, player_index: int):
    ps = obs.current.players[player_index]
    match area:
        case AreaType.DECK:
            return obs.select.deck[index]
        case AreaType.HAND:
            return ps.hand[index]
        case AreaType.DISCARD:
            return ps.discard[index]
        case AreaType.ACTIVE:
            return ps.active[index]
        case AreaType.BENCH:
            return ps.bench[index]
        case AreaType.PRIZE:
            return ps.prize[index]
        case AreaType.STADIUM:
            return obs.current.stadium[index]
        case AreaType.LOOKING:
            return obs.current.looking[index]
        case _:
            return None


# ---- decoder 特征（逐候选动作）---------------------------------------------
def decoder_main(sv: SparseVector, feature_index: int, card):
    if card is not None:
        sv.add(decoder_card_offset + feature_index * card_count + card.id, 1)


def decoder_card_id(sv: SparseVector, context: SelectContext, card_id: int):
    sv.add(decoder_card_offset + (decoder_main_feature + context) * card_count + card_id, 1)


def decoder_card(sv: SparseVector, context: SelectContext, card):
    if card is not None:
        decoder_card_id(sv, context, card.id)


def get_decoder_input(obs: Observation, actions: list[list[int]]) -> SparseVector:
    sv = SparseVector()
    your_index = obs.current.yourIndex
    ps = obs.current.players[your_index]
    context = obs.select.context
    for action in actions:
        sv.word_start()
        if len(action) == 0:
            sv.add(0, 1)
            continue
        for i in action:
            o = obs.select.option[i]
            match o.type:
                case OptionType.END:
                    sv.add(1, 1)
                case OptionType.YES:
                    sv.add(2, 1)
                case OptionType.NO:
                    sv.add(3, 1)
                case OptionType.SPECIAL_CONDITION:
                    sv.add(4 + o.specialConditionType, 1)
                case OptionType.NUMBER:
                    sv.add(9 + min(o.number, 4), 1)
                case OptionType.ATTACK:
                    sv.add(decoder_attack_offset + o.attackId, 1)
                case OptionType.PLAY:
                    decoder_main(sv, 0, ps.hand[o.index])
                case OptionType.ATTACH:
                    decoder_main(sv, 1, get_card(obs, o.area, o.index, your_index))
                    decoder_main(sv, 2, get_card(obs, o.inPlayArea, o.inPlayIndex, your_index))
                case OptionType.EVOLVE:
                    decoder_main(sv, 3, get_card(obs, o.area, o.index, your_index))
                    decoder_main(sv, 4, get_card(obs, o.inPlayArea, o.inPlayIndex, your_index))
                case OptionType.ABILITY:
                    decoder_main(sv, 5, get_card(obs, o.area, o.index, your_index))
                case OptionType.DISCARD:
                    decoder_main(sv, 6, get_card(obs, o.area, o.index, your_index))
                case OptionType.RETREAT:
                    decoder_main(sv, 7, ps.active[0])
                case OptionType.CARD:
                    decoder_card(sv, context, get_card(obs, o.area, o.index, o.playerIndex))
                case OptionType.TOOL_CARD:
                    card = get_card(obs, o.area, o.index, o.playerIndex)
                    decoder_card(sv, context, card.tools[o.toolIndex])
                case OptionType.ENERGY_CARD | OptionType.ENERGY:
                    card = get_card(obs, o.area, o.index, o.playerIndex)
                    decoder_card(sv, context, card.energyCards[o.energyIndex])
                case OptionType.SKILL:
                    decoder_card_id(sv, context, o.cardId)
    return sv


def enumerate_actions(obs: Observation, cap: int = 64) -> list[list[int]]:
    """枚举当前可选的「动作」（option 下标的组合），最多 cap 个。

    单选时就是每个 option；多选(maxCount>1)时枚举组合。移植自样例 create_node。
    """
    actions: list[list[int]] = []
    max_count = obs.select.maxCount
    n = len(obs.select.option)
    indices = list(range(max_count))
    for _ in range(cap):
        actions.append(indices.copy())
        for i in range(len(indices)):
            index = len(indices) - i - 1
            if indices[index] < n - i - 1:
                indices[index] += 1
                for j in range(index + 1, len(indices)):
                    indices[j] = indices[j - 1] + 1
                break
        else:
            break
    return actions
