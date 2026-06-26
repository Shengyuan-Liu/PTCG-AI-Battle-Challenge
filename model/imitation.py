"""SL warm-start：从 replay 做行为克隆，给 self-play 一个更好的起点。

官方样例直接从随机权重 self-play（早期数据质量差、爬得慢）。我们多一步：先在下载的
强对局 replay 上模仿专家动作，把网络预热到一个像样的策略，再交给 selfplay.train
（--init-weights）继续 RL。复用同一套 encoding/network/训练循环。

样本构造（每个「玩家做了选择」的 step）：
  sv_enc = get_encoder_input(obs, 该玩家的牌表)
  sv_dec = get_decoder_input(obs, enumerate_actions(obs))
  policy 标签：专家实际选的那个动作 = +1，其余 = -1（softmax 后专家动作概率高）
  value  标签：该玩家本局最终胜负（+1/-1/0）
"""
import argparse
import glob
import json
import os

import torch

from .encoding import enumerate_actions, get_decoder_input, get_encoder_input
from .mcts import LearnSample
from .network import new_model
from .selfplay import train_on

from cg.api import to_observation_class

DECK_LEN = 60


def _player_decks(steps) -> dict:
    """从开局的两个选卡组动作（len==60）取双方牌表 {player_index: deck}。"""
    decks = {}
    for step in steps:
        for pi, player in enumerate(step):
            act = player.get("action")
            if isinstance(act, list) and len(act) == DECK_LEN and pi not in decks:
                decks[pi] = act
    return decks


def _match_action(obs, chosen: list[int]) -> int:
    """专家选的 option 下标列表 → 在 enumerate_actions 里的位置；找不到返回 -1。"""
    target = sorted(chosen)
    for k, a in enumerate(enumerate_actions(obs)):
        if sorted(a) == target:
            return k
    return -1


def samples_from_replay(path: str) -> list[LearnSample]:
    try:
        d = json.load(open(path, encoding="utf-8"))
    except Exception:
        return []
    rewards = d.get("rewards", [0, 0])
    steps = d.get("steps", [])
    decks = _player_decks(steps)
    out = []
    for step in steps:
        for pi, player in enumerate(step):
            act = player.get("action")
            if not act or len(act) == DECK_LEN or pi not in decks:
                continue
            obs_dict = player.get("observation")
            if not obs_dict or obs_dict.get("current") is None or obs_dict.get("select") is None:
                continue
            try:
                obs = to_observation_class(obs_dict)
                actions = enumerate_actions(obs)
                k = _match_action(obs, act)
                if k < 0 or k >= len(actions):
                    continue
                sv_enc = get_encoder_input(obs, decks[pi])
                sv_dec = get_decoder_input(obs, actions)
                value = float(rewards[pi]) if pi < len(rewards) else 0.0
                policy = [1.0 if j == k else -1.0 for j in range(len(actions))]
                out.append(LearnSample(value, policy, sv_enc, sv_dec))
            except Exception:
                continue            # 单条脏数据不影响整体
    return out


def build_dataset(replay_dir: str = "data/replay", limit: int | None = None) -> list[LearnSample]:
    samples = []
    files = sorted(glob.glob(os.path.join(replay_dir, "*.json")))
    for i, path in enumerate(files):
        samples.extend(samples_from_replay(path))
        if limit and len(samples) >= limit:
            break
        if (i + 1) % 50 == 0:
            print(f"  parsed {i+1}/{len(files)} files, {len(samples)} samples", flush=True)
    return samples


def pretrain(replay_dir="data/replay", epochs=3, only_winners=True,
             out_path="model/out/sl_init.pth"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = new_model(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
    samples = build_dataset(replay_dir)
    if only_winners:
        samples = [s for s in samples if s.value > 0]      # 只学赢家（replay 里弱 agent 多）
    print(f"行为克隆样本数：{len(samples)}（only_winners={only_winners}）device={device}")
    if not samples:
        print("没有可用样本——先用 tools/download_replays.py 下载 replay。")
        return
    for ep in range(epochs):
        train_on(model, optimizer, samples, device)
        print(f"[epoch {ep}] done", flush=True)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    torch.save(model.state_dict(), out_path)
    print(f"已保存 SL 预热权重 → {out_path}")
    print(f"下一步：python -m model.selfplay --init-weights {out_path}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--replay-dir", default="data/replay")
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--all", action="store_true", help="不止学赢家，全部都学")
    ap.add_argument("--out", default="model/out/sl_init.pth")
    a = ap.parse_args()
    pretrain(a.replay_dir, a.epochs, not a.all, a.out)
