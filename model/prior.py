"""从 replay 抽「对手卡组先验」——改进 MCTS 隐藏信息 determinization 的数据基础。

官方样例对看不到的对手牌库/手牌/奖赏全填常量（Snorlax / 基础能量，注释明说 no deep
meaning），导致搜索对对手能力的预测严重失真。我们用真实数据改进：

replay 每局开局有两个「选卡组」动作（action 长度 == 60，即双方完整 60 张牌表）。
扫描 data/replay/*.json 收集这些真实牌表，得到：
  - decklists : 真实 60 张牌表列表（可整副采样一个「像样的对手卡组」）
  - card_freq : 各 cardId 在牌表中出现的频次（加权采样的 fallback）
结果缓存到 data/replay_prior.json，避免每次重扫。

无 replay 时返回空先验，调用方自动退回常量填充（保证永远能跑）。
"""
import glob
import json
import os
from collections import Counter

CACHE = "data/replay_prior.json"
DECK_LEN = 60


def _iter_decklists(replay_dir: str):
    for path in glob.glob(os.path.join(replay_dir, "*.json")):
        try:
            d = json.load(open(path, encoding="utf-8"))
        except Exception:
            continue
        for step in d.get("steps", []):
            for player in step:
                act = player.get("action")
                # 选卡组动作：恰好 60 个 card ID（不是下标）
                if isinstance(act, list) and len(act) == DECK_LEN and all(isinstance(x, int) for x in act):
                    yield act


def build_prior(replay_dir: str = "data/replay", save: bool = True) -> dict:
    decklists, freq = [], Counter()
    seen = set()
    for deck in _iter_decklists(replay_dir):
        key = tuple(sorted(deck))
        if key in seen:                 # 同一副牌表去重（双方各记一次即可）
            continue
        seen.add(key)
        decklists.append(deck)
        freq.update(deck)
    prior = {
        "decklists": decklists,
        "card_freq": {str(k): v for k, v in freq.items()},
        "num_decks": len(decklists),
    }
    if save and decklists:
        os.makedirs(os.path.dirname(CACHE), exist_ok=True)
        json.dump(prior, open(CACHE, "w"), separators=(",", ":"))
    return prior


def load_prior(replay_dir: str = "data/replay") -> dict:
    """优先读缓存；没有则现扫；都没有则返回空先验。"""
    if os.path.exists(CACHE):
        try:
            return json.load(open(CACHE, encoding="utf-8"))
        except Exception:
            pass
    return build_prior(replay_dir)


class OpponentPrior:
    """采样对手未知卡的辅助器。"""
    def __init__(self, prior: dict):
        self.decklists = prior.get("decklists", []) if prior else []
        cf = (prior or {}).get("card_freq", {})
        self.cards = [int(k) for k in cf]
        self.weights = [cf[k] for k in cf]

    @property
    def available(self) -> bool:
        return bool(self.decklists)

    def sample_cards(self, n: int, rng) -> list[int]:
        """采 n 张「像真实卡组里会出现」的卡。

        优先：随机取一副真实牌表，从中无放回采样（最真实）；
        牌表不够大时用频次加权有放回补齐。
        """
        if not self.available or n <= 0:
            return []
        deck = rng.choice(self.decklists)
        if n <= len(deck):
            return rng.sample(deck, n)
        # 牌表不够 → 频次加权补齐
        extra = rng.choices(self.cards, weights=self.weights, k=n - len(deck))
        return list(deck) + extra


if __name__ == "__main__":
    p = build_prior()
    print(f"decklists={p['num_decks']}  distinct_cards={len(p['card_freq'])}")
    if p["card_freq"]:
        top = sorted(p["card_freq"].items(), key=lambda kv: -kv[1])[:10]
        print("top cards:", top)
