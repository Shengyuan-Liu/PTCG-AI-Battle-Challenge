# last_submission.md — 最近一次提交台账

> **规则:每次 `kaggle competitions submit` 之后，必须立刻更新本文件**（模型、训练方式、数据、评估、天梯表现、复现命令）。这是提交流程的强制步骤，见 CLAUDE.md。

---

## 当前最近一次提交：v5

| 项 | 值 |
|---|---|
| 提交号 (ref) | 54093633 |
| 文件 | submission.tar.gz（105 MiB） |
| 时间 | 2026-06-27 02:09 UTC |
| 描述 | big Transformer policy (29M, 60% vs rule), v5 |
| 验证状态 | COMPLETE |
| **天梯分** | **501.3（起步 μ₀=600，3h 内掉到 501 → 实际弱于 rule 的 ~544）** |

### 模型
- **类型**：纯 NN 策略 agent（**推理时单次前向选招，无 MCTS**；`model/agent.py` 的 `USE_MCTS=False` → `_policy_move`）。
- **架构**：小 Transformer encoder-decoder（`model/network.py`，`ARCH`）
  - `d_model=256, num_heads=4, d_feedforward=1024, num_layers_encoder=3, num_layers_decoder=3` ≈ **29M 参数**，权重 117MB。
  - encoder: `EmbeddingBag(sum)` 稀疏特征 → TransformerEncoder → value 头 `tanh(mean)`。
  - decoder: 候选动作 cross-attention → 每动作一个 policy 标量。
- **输入编码**：`model/encoding.py`（Observation → 稀疏 EmbeddingBag 特征；候选动作 = `enumerate_actions`）。
- **输出**：对当前合法动作的 policy 打分 → argmax 选招（masked）。

### 训练方式
1. **SL warm-start**（`model/imitation.py`）：行为克隆，专家动作 one-hot，只学赢家。**1135 个样本**，4 epoch，→ `sl_init.pth`。
2. **AZ self-play**（`model/parallel_selfplay.py --mode az`）：
   - 标准 AlphaZero：policy 标签 = MCTS 访问次数分布 + 交叉熵；value 标签 = 最终胜负 z。
   - **SL 锚**：每轮训练混入 1135 个 replay 行为克隆样本，防遗忘。
   - 超参：`lr=1e-4, searches=20, workers=16, selfplay-games=64, eval-games=48, patience=5`。
   - 结果：best **69% vs rule @ iter6**（早停），→ `model/out_big/model_final.pth`。
   - **注意**：self-play 自造数据是主训练信号，不受 replay 数量限制（这是大模型在数据少时还能训的原因，但也导致只跟自己/单一 rule 对练 → 过拟合）。
3. 训练环境：本机 RTX 4080 Laptop（GPU 训练）+ 32 核（16 worker 并行 self-play，CPU 推理）。

### 数据
- **32 个 Kaggle replay**（`data/replay/*.json`，官方每日 top 局数据集的子集）。
- 对手卡组先验：从 replay 开局选卡组动作抽出 **19 副真实牌表 / 107 种卡**（`model/prior.py` → `data/replay_prior.json`）。
- 卡组：`sample_submission/deck.csv`（60 张）。

### 评估
- **本地 arena（纯 NN，对 rule，200 局）**：大模型 **60%** vs 小模型 v4 **52%** → 本地看更强。
- **⚠️ 天梯打脸**：v5=501 < rule≈544。**"vs 自家 rule" 是误导性代理指标**——rule 本身低于平均(544<600)，且 self-play 在 32 局上过拟合，泛化差。**天梯才是最终裁判**。

### 提交包结构
```
submission/ → submission.tar.gz
├── main.py            # 入口：模型优先 + rule 兜底；顶层不用 __file__（kaggle exec 无 __file__）
├── deck.csv  rule_agent.py
├── cg/ (libcg.so)     # Linux 引擎
└── model/{__init__,agent,encoding,network}.py + out/model_final.pth(=大模型)
```

### 复现命令
```bash
export KAGGLE_API_TOKEN=KGAT_...
python -m model.prior                                   # 建先验
python -m model.imitation --epochs 4                    # SL → sl_init.pth
python -m model.parallel_selfplay --mode az --init-weights model/out/sl_init.pth \
    --workers 16 --selfplay-games 64 --eval-games 48 --searches 20 --lr 1e-4 \
    --eval-opponent rule --patience 5 --iterations 25 --out-dir model/out_big
# 打包：见 §提交包结构；本地用 exec-无-__file__ 测试后 tar -czf
```

### 已知问题 / 下一步
- **天梯弱于 rule**：根因 = 数据太少(32 局) + self-play 过拟合 + 缺对手多样性。
- **最高优先**：多拉真实 replay（dataset API 列文件接口当日限流，待重置）→ 强 SL + 多样对手池。
- 数据不足时，29M 大模型过拟合反而比小模型差；要么先补数据再放大，要么先用 rule 占天梯位。

---

## 提交历史（简表）

| ref | ver | 模型 | 天梯分 | 备注 |
|---|---|---|---|---|
| 54093633 | v5 | 大 Transformer 29M（纯NN） | 501 | 本地 60% vs rule，但天梯弱于 rule |
| 54091919 | v4 | 小 NN 12M（纯NN） | ~500 | __file__ bug 修复后首个 COMPLETE 的 NN |
| 54084770 | v3 | rule-based（无 torch） | ~544 | 安全基线，当前最强天梯选手 |
| 54084685 | v2 | 纯 NN + rule 兜底 | ERROR | `__file__` exec bug |
| 54084083 | v1 | MCTS+NN + rule 兜底 | ERROR | `__file__` exec bug |
| 53975678 | v0 | rule-based | 544 | 最早的 rule 提交 |
