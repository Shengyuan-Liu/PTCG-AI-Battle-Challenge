"""AlphaZero 式 self-play 训练循环。移植自官方样例，改动：
  - 用项目自己的 deck.csv（可传入）；
  - MCTS 用数据驱动 prior（mcts.determinize）；
  - 支持从已有权重（如 SL warm-start）续训。
"""
import argparse
import csv
import os
import random
import sys

import torch

from .mcts import LearnSample, mcts_agent
from .network import LearnInput, MyModel, new_model
from .prior import OpponentPrior, load_prior

# cg 引擎相关延迟到运行时再 import（Mac 上 import 即失败）
from cg.api import to_observation_class
from cg.game import battle_finish, battle_select, battle_start


def load_deck(path: str = "sample_submission/deck.csv") -> list[int]:
    with open(path, newline="") as f:
        return [int(row[0]) for row in csv.reader(f) if row and row[0].strip().isdigit()]


def random_agent(obs_dict: dict) -> list[int]:
    obs = to_observation_class(obs_dict)
    n = len(obs.select.option)
    return random.sample(range(n), min(obs.select.maxCount, n)) if n else []


def evaluate(model, deck, prior, games: int, searches: int) -> float:
    """对随机对手胜率（不含平局的胜率）。"""
    model.eval()
    win = lose = 0
    with torch.inference_mode():
        for g in range(games):
            obs, _ = battle_start(deck, deck)
            yi = g % 2
            while obs["current"]["result"] < 0:
                if obs["current"]["yourIndex"] == yi:
                    sel, _ = mcts_agent(obs, deck, model, prior, num_searches=searches)
                else:
                    sel = random_agent(obs)
                obs = battle_select(sel)
            r = obs["current"]["result"]
            battle_finish()
            if r == yi:
                win += 1
            elif r != 2:
                lose += 1
    return 100 * win / max(win + lose, 1)


def collect_selfplay(model, deck, prior, games: int, searches: int) -> list[LearnSample]:
    """自对弈收集训练样本（含 TD(λ) value 标签）。"""
    model.eval()
    out: list[LearnSample] = []
    LAMBDA = 0.9
    with torch.inference_mode():
        for _ in range(games):
            obs, _ = battle_start(deck, deck)
            samples = [[], []]
            while obs["current"]["result"] < 0:
                sel, sample = mcts_agent(obs, deck, model, prior, num_searches=searches)
                if sample is not None:
                    samples[obs["current"]["yourIndex"]].append(sample)
                obs = battle_select(sel)
            result = obs["current"]["result"]
            battle_finish()
            for i in range(2):
                value = 1.0 if i == result else -1.0
                for sample in reversed(samples[i]):
                    sample.value, value = (value + sample.value) * 0.5, value * LAMBDA + sample.value * (1 - LAMBDA)
                    out.append(sample)
    return out


def train_on(model, optimizer, samples: list[LearnSample], device, batch_size: int = 128):
    loss_fn_enc = torch.nn.HuberLoss(delta=0.2)
    loss_fn_dec = torch.nn.HuberLoss(reduction="none", delta=0.1)
    model.train()
    random.shuffle(samples)
    for i in range(len(samples) // batch_size):
        in_enc, in_dec = LearnInput(), LearnInput()
        mask, label_enc, label_dec = [], [], []
        for s in samples[batch_size * i: batch_size * (i + 1)]:
            in_enc.add(s.sv_enc)
            in_dec.add(s.sv_dec)
            label_enc.append(s.value)
            label_dec.extend(s.policy)
            mask.extend([1.0] * len(s.policy))
            for _ in range(64 - len(s.policy)):                 # pad 到 64
                mask.append(0.0); label_dec.append(0.0)
                in_dec.offset.append(len(in_dec.index))
        mt = torch.tensor(mask, dtype=torch.float32, device=device).view(batch_size, -1)
        le = torch.tensor(label_enc, dtype=torch.float32, device=device).view(batch_size, -1)
        ld = torch.tensor(label_dec, dtype=torch.float32, device=device).view(batch_size, -1)
        optimizer.zero_grad()
        out_enc, out_dec = model(
            torch.tensor(in_enc.index, dtype=torch.int32, device=device),
            torch.tensor(in_enc.value, dtype=torch.float32, device=device),
            torch.tensor(in_enc.offset, dtype=torch.int32, device=device),
            torch.tensor(in_dec.index, dtype=torch.int32, device=device),
            torch.tensor(in_dec.value, dtype=torch.float32, device=device),
            torch.tensor(in_dec.offset, dtype=torch.int32, device=device))
        loss = loss_fn_enc(out_enc, le) + (loss_fn_dec(out_dec, ld) * mt).sum() / batch_size
        loss.backward()
        optimizer.step()


def train_on_az(model, optimizer, samples, device, batch_size: int = 128):
    """标准 AlphaZero 损失：value Huber + policy 交叉熵（masked，与推理一致的 *10 缩放）。

    samples 的 .policy 是「该样本各动作上的概率分布」（和为 1）：
      self-play 样本 = MCTS 访问次数分布；SL 锚样本 = 专家动作 one-hot。
    """
    import torch.nn.functional as F
    loss_fn_v = torch.nn.HuberLoss(delta=0.2)
    model.train()
    random.shuffle(samples)
    for i in range(len(samples) // batch_size):
        in_enc, in_dec = LearnInput(), LearnInput()
        mask, label_v, target_p = [], [], []
        for s in samples[batch_size * i: batch_size * (i + 1)]:
            in_enc.add(s.sv_enc)
            in_dec.add(s.sv_dec)
            label_v.append(s.value)
            tot = sum(s.policy) or 1.0
            target_p.extend(p / tot for p in s.policy)       # 归一化成分布
            mask.extend([1.0] * len(s.policy))
            for _ in range(64 - len(s.policy)):
                mask.append(0.0); target_p.append(0.0)
                in_dec.offset.append(len(in_dec.index))
        mt = torch.tensor(mask, dtype=torch.float32, device=device).view(batch_size, -1)
        lv = torch.tensor(label_v, dtype=torch.float32, device=device).view(batch_size, -1)
        tp = torch.tensor(target_p, dtype=torch.float32, device=device).view(batch_size, -1)
        optimizer.zero_grad()
        out_enc, out_dec = model(
            torch.tensor(in_enc.index, dtype=torch.int32, device=device),
            torch.tensor(in_enc.value, dtype=torch.float32, device=device),
            torch.tensor(in_enc.offset, dtype=torch.int32, device=device),
            torch.tensor(in_dec.index, dtype=torch.int32, device=device),
            torch.tensor(in_dec.value, dtype=torch.float32, device=device),
            torch.tensor(in_dec.offset, dtype=torch.int32, device=device))
        logits = (out_dec * 10.0).masked_fill(mt == 0, -1e9)  # 与 create_node 的 exp(p*10) 一致
        logp = F.log_softmax(logits, dim=1)
        loss_p = -(tp * logp).sum(dim=1).mean()
        loss = loss_fn_v(out_enc, lv) + loss_p
        loss.backward()
        optimizer.step()


def to_onehot_policy(sample):
    """把 SL 的 [+1/-1] advantage 标签转成 one-hot 分布（给 AZ 交叉熵当 SL 锚）。"""
    k = max(range(len(sample.policy)), key=lambda j: sample.policy[j])
    sample.policy = [1.0 if j == k else 0.0 for j in range(len(sample.policy))]
    return sample


def train(iterations=5, eval_games=20, selfplay_games=40, searches=10,
          deck_path="sample_submission/deck.csv", init_weights=None, out_dir="model/out"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = new_model(device)
    if init_weights and os.path.exists(init_weights):
        model.load_state_dict(torch.load(init_weights, map_location=device))
        print(f"loaded init weights from {init_weights}")
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
    deck = load_deck(deck_path)
    prior = OpponentPrior(load_prior())
    os.makedirs(out_dir, exist_ok=True)
    print(f"device={device} deck={len(deck)} prior={'data' if prior.available else 'constant'} ")

    for it in range(iterations):
        wr = evaluate(model, deck, prior, eval_games, searches)
        print(f"[iter {it}] win rate vs random = {wr:.0f}%", flush=True)
        torch.save(model.state_dict(), os.path.join(out_dir, f"model{it}.pth"))
        samples = collect_selfplay(model, deck, prior, selfplay_games, searches)
        train_on(model, optimizer, samples, device)
        print(f"[iter {it}] trained on {len(samples)} samples", flush=True)
    wr = evaluate(model, deck, prior, eval_games, searches)
    print(f"[final] win rate vs random = {wr:.0f}%", flush=True)
    torch.save(model.state_dict(), os.path.join(out_dir, "model_final.pth"))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--iterations", type=int, default=5)
    ap.add_argument("--eval-games", type=int, default=20)
    ap.add_argument("--selfplay-games", type=int, default=40)
    ap.add_argument("--searches", type=int, default=10)
    ap.add_argument("--deck", default="sample_submission/deck.csv")
    ap.add_argument("--init-weights", default=None, help="续训权重（如 SL warm-start 产物）")
    ap.add_argument("--out-dir", default="model/out")
    a = ap.parse_args()
    train(a.iterations, a.eval_games, a.selfplay_games, a.searches, a.deck, a.init_weights, a.out_dir)
