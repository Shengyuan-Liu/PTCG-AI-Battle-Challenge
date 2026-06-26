# model.md — PTCG AI Battle 模型设计文档

本文件描述这个项目的核心 AI 模型：**它是什么、输入输出长什么样、用什么架构、拿什么数据怎么训、推理时怎么用**。是策略路线（CLAUDE.md §7）里「搜索 + RL」阶段的落地规格。语言以中文为主，技术名词 / API 保留英文。

> 维护提示：本文件是「模型记忆」。架构、特征工程、训练流程、超参有大改动时更新。术语与引擎 API 以 [CLAUDE.md](../CLAUDE.md) §6 为准。
>
> 最近更新：2026-06-27（基于官方 RL+MCTS 样例落地 `model/` 实现，含两项改进；见 §9。确认引擎在本机 Windows 可跑，能本地训练/自对弈）。
> 首版：2026-06-26（确定 SL→PPO→AlphaZero 路线与小 Transformer 架构）。

---

## 0. TL;DR

- **模型本体** = 一个 **policy + value 双头神经网络**（小、CPU 友好）。
- **角色** = 给 **MCTS 搜索**当先验(policy)和估值(value)。真正的「前向推演 / 推理」由搜索完成，网络只是让搜索更准更快。
- **架构** = 「实体编码 + 聚合 + 双头 + action masking」。基线用 **DeepSets**（每实体小 MLP + 池化），加强版把池化升级成 **1–2 层小 Transformer 自注意力**。**不用 CNN**（非网格数据），**不用大 Transformer / LLM**（CPU + 10min + MCTS 调用频繁会超时）。
- **训练** = ① **SL 行为克隆**（模仿 replay）→ ② **maskable PPO**（胜负=可验证奖励）→ ③ **AlphaZero 式 self-play + MCTS** 闭环。
- **数据** = Kaggle 官方每日打包的 episode replay（`tools/download_replays.py` 下载），拆成 `(observation, action)` 样本。

---

## 1. 模型定位：它在系统里是什么

```
        MCTS(搜索, 推理时跑) ──产出更强的走法分布──►  当 RL 训练目标
            ▲                                          │
            │ 网络提供 policy 先验 + value 估值          ▼
        policy+value 网络  ◄────── RL/SL 训练 ──────────┘
```

- **不是**「一次前向直接出动作」的纯分类器那么简单，也**不是** LLM 那种生成思维链的模型。
- 它是 **AlphaZero 范式**里的那张网：
  - **policy 头** → 告诉搜索「优先试哪几个动作」（剪枝）；
  - **value 头** → 告诉搜索「这个局面赢面多少」（不必模拟到底）。
- **推理能力 ≈ 搜索深度 × 网络先验/估值质量**。所以网络要小（让搜索能多跑几次），结构要对（抓住卡牌游戏的实体/关系）。

> 详细的「为什么 RL 不是在强化思维链、推理来自搜索」见 CLAUDE.md §7 与本文件 §6/§7。

---

## 2. 模型架构

### 2.1 总览

```
卡牌ID ──► Embedding 表 (cardId → 向量, dim≈16–32)        ← 卡池固定，可学嵌入；attackId/技能可同法嵌入
                │
每个「实体」(场上每只宝可梦 / 关注的每张牌) 的特征向量：
  [卡嵌入  +  hp/maxHp 归一  +  各 EnergyType 能量计数  +  特殊状态 flags
   +  appearThisTurn  +  tools 嵌入  +  preEvolution 信息 ...]
                │   每实体一个共享的小 MLP (entity encoder)
                ▼
   ┌──────────────── 聚合 aggregate ────────────────┐
   │ 基线版  : DeepSets —— 对实体集合做 sum/mean 池化   │  ← 先做这个
   │ 加强版  : 1–2 层自注意力 (小 Transformer)         │  ← 后升级
   │           分区做(我方active/bench、对方active/bench)│
   └────────────────────────────────────────────────┘
                │
   +  全局特征 (见 §3.3：回合、奖赏差、牌库数、手牌 bag-of-cards、场地、先后手、各 flag)
                ▼
            Trunk MLP (2–3 层) ──► 状态嵌入 h  (dim≈128–256)
                │
        ┌───────┴────────────────────────────────┐
   Policy 头                                   Value 头
 对每个「合法 option」:                          h → MLP → tanh
   option 特征 + 引用卡/区域嵌入 + h               → 标量 ∈ [-1,1]  (本方视角赢面)
   → 小 MLP → 1 个 logit
 → 仅在合法 option 上 softmax (action masking)
```

### 2.2 为什么这么选（架构对比）

| 方案 | 结论 |
|---|---|
| 纯 MLP（把一切拍平成定长向量） | 能跑但丢失「实体集合」结构，关系推理弱。仅作 fallback/对照。 |
| **DeepSets（实体小MLP + 池化）** | ⭐ **推荐基线**：处理变长集合、置换不变、CPU 飞快、实现简单。 |
| **小 Transformer（实体当 token，自注意力）** | ⭐ **加强版**：卡牌游戏的天然正确结构（集合 + 关系/克制推理）。务必小（d_model 小、层少、token 仅几十个）。 |
| CNN | ❌ observation 非网格，不适用。 |
| 大 Transformer / LLM | ❌ CPU + 10min + MCTS 高频调用 → 超时判负。 |

### 2.3 规模约束（硬指标）

- **CPU 推理、单次前向毫秒级**（MCTS 每步要调几百~上千次）。
- **参数量小**，权重要能打进 `submission.tar.gz`（无外网，权重随包）。
- 初始建议：embedding dim 16–32，trunk hidden 128–256，1–3 层；先小后大，以「搜索能多跑」为准绳。

---

## 3. 输入：Observation → 张量

输入来自 agent 每步收到的 `obs_dict`，经 `to_observation_class()` 转成 dataclass（结构见 CLAUDE.md §6.1）。把这棵树编码成网络可吃的张量，分三类特征。

### 3.1 实体特征（变长集合，喂给 entity encoder）

对每个 `Pokemon`（我方/对方的 active、bench）生成一个特征向量：

- `id` → 卡 Embedding；
- `hp / maxHp`（归一化）、是否濒死；
- `energies: list[EnergyType]` → 按 11 种 EnergyType 计数的定长向量；
- 特殊状态：poison/burn/asleep/paralyze/confuse（仅 active 有）→ flags；
- `appearThisTurn`（刚上场→不能攻击/进化）；
- `tools` → 工具卡嵌入；`preEvolution` → 进化链信息；
- 可附：该宝可梦可用 attack 的 `energies` 需求是否已满足（用 `remainEnergyCost` / 能量对比预算）。

> 暗置（None）的实体用一个「未知」占位向量；对手暗置 active 在搜索阶段由预测填充（见 §7 隐藏信息）。

### 3.2 卡集合特征（用「按卡池计数」绕开变长）

卡池**固定**，所以这些区用 **bag-of-cards 定长计数向量**（长度 = 卡池大小，或按需做降维/嵌入求和）表示：

- 我方 `hand`（`PlayerState.hand` 我方可见）→ 每张卡计数；
- 我方/对方 `discard`、`deckCount`、`prize`（暗置=None 只用计数）；
- 对手 `hand` 不可见 → 只有 `handCount`（标量），内容在搜索阶段采样填充。

### 3.3 全局/标量特征（拼到 trunk 前）

- `State.turn`（含先后手相位）、`turnActionCount`、`firstPlayer`、`yourIndex`；
- 每回合一次性资源 flags：`supporterPlayed` / `stadiumPlayed` / `energyAttached` / `retreated`；
- **奖赏差**（双方 `prize` 剩余数之差，核心估值信号）；
- 双方 `deckCount`（deck-out 风险）、`benchMax`、`stadium`、`looking`；
- 当前 `SelectData` 的语义：`type`(SelectType) / `context`(SelectContext) / `minCount` / `maxCount` / `remainDamageCounter` / `remainEnergyCost` —— 让网络知道「现在在选什么、要选几个」。

### 3.4 候选动作特征（policy 头逐个打分用）

当前合法动作 = `obs.select.option` 列表（变长）。对每个 option 按其 `OptionType`（CLAUDE.md §6.2）抽特征：

- 通用：OptionType one-hot；
- 引用的卡/区域：`area`(AreaType) / `index` / `playerIndex` → 对应实体嵌入或区域嵌入；
- ATTACK(13)：`attackId` → attack 嵌入（含 damage/energies，来自 `all_attack()`）；
- PLAY(7)/ATTACH(8)/EVOLVE(9)/ABILITY(10)/RETREAT(12)/END(14)/SKILL(15)/SPECIAL_CONDITION(16) 等各取其字段。

> 对未知/新增的 Enum 值要**容错**（比赛期间可能新增），缺字段给默认嵌入，绝不崩溃。

---

## 4. 输出：动作

### 4.1 网络输出

- **Policy**：对当前 N 个合法 option 各产一个 logit → **masked softmax**（只在合法集合上归一）。即「逐选项打分 + masking」的 pointer 式输出，**输出维度每步随 N 变化**。
- **Value**：标量 ∈ [-1, 1]，本方视角的预期胜负（+1 赢 / -1 输），对齐 `cabt.json` 的 reward（CLAUDE.md §6.6）。

### 4.2 映射回提交契约

Agent 契约是 `agent(obs_dict) -> list[int]`（CLAUDE.md §5）：

- **常规选择**：返回 `obs.select.option` 的**下标列表**，长度 ∈ `[minCount, maxCount]`、元素不重复、不越界。
  - 单选（maxCount=1）：取 policy argmax / 采样的那个下标。
  - 多选（maxCount>1，如弃牌、贴多能量）：按 policy 分数贪心取 top-k 直到满足约束；或在搜索里逐步决定。
  - 可空（minCount=0）：允许返回 `[]` 表示跳过。
- **初始选卡组**：`obs.select is None and obs.current is None` 时，返回**固定 60 张 card ID**（不是下标）——这步**不学**，直接用预设 `deck.csv`。
- **健壮性底线**：任何分支都返回合法动作，超时/异常兜底**随机合法选择**，绝不崩溃（CLAUDE.md §5）。

> 数据印证（实测 `data/replay/81729405.json`）：step 1 玩家 0 的 action 是 60 个 card ID（选卡组）；其余 step 的 action 是 `[1]`/`[0]`/`[4]` 这种**单/多下标列表**。

---

## 5. 训练数据

### 5.1 来源与下载

- 官方每天把 **top 局**（按参与者平均评分筛过）打包成数据集 `kaggle/pokemon-tcg-ai-battle-episodes-<date>`；`kaggle/pokemon-tcg-ai-battle-episodes-index` 的 `manifest.csv` 列出有哪些天（每天约 5000–7800 局、~20GB）。
- 下载工具：**[`tools/download_replays.py`](../tools/download_replays.py)**。用法：
  ```bash
  export KAGGLE_API_TOKEN=KGAT_xxx
  kaggle datasets download kaggle/pokemon-tcg-ai-battle-episodes-index -p data/episodes-index --unzip
  python tools/download_replays.py --latest 1 --limit 300      # 控量
  python tools/download_replays.py --date 2026-06-25 --limit 0 # 整天(~20GB)
  ```
- 落地：`data/replay/<EpisodeId>.json`（已 gitignore，不进仓库）。
- ⚠️ Kaggle「列文件」接口限流较狠；脚本已做**文件名缓存 + 退避**，正常一天跑一两次不会触发。

### 5.2 replay JSON 结构（实测）

顶层 key：`steps` / `rewards` / `info` / `configuration` / ...

- `rewards`：`[r0, r1]`，胜 +1 / 负 -1 / 平 0（对应玩家 0/1）。
- `info.Agents`：两个玩家名（可用于按对手强弱筛选）。
- `steps`：长度 = 总步数；**每个元素是 list[2]**（两个玩家各一份），每份含：
  - `observation`：keys = `current` / `select` / `logs` / `step` / `remainingOverageTime` / `search_begin_input`（即 agent 当步收到的 obs）；
  - `action`：该玩家这步交的动作（下标列表，或选卡组的 60 个 card ID）；非空表示「轮到他做选择」；
  - `status`：`ACTIVE` / `INACTIVE` 等。

### 5.3 提取训练样本（SL 用）

遍历每个 step、每个玩家，挑「`action` 非空」的：得到一条样本

```
x = encode(observation)          # §3 的张量
y = action                       # §4 的下标列表（标签）
weight/return = rewards[player]   # 该局该玩家最终胜负
```

清洗规则：

- **丢掉选卡组那步**（action=60 card IDs，不学，用预设卡组）。
- **只学赢家**（或给赢家样本更高权重）——replay 里弱 agent 多，模仿全部=学垃圾。可按 `info.Agents` / 评分进一步筛强对局。
- 多选/特殊状态 step 的 label 处理见 §4.2。
- 对未知 Enum 值容错跳过，不让单条脏数据中断整批。

### 5.4 self-play 数据（RL 用）

阶段③不依赖外部 replay：用当前网络 + MCTS **自对弈**生成 `(observation, MCTS 访问分布 π, 最终胜负 z)`，作为 AlphaZero 式训练目标。对手池 = 历史版本网络 + rule-based（防过拟合到单一对手）。

---

## 6. 训练流程（SL → PPO → AlphaZero）

### 阶段① SL / 行为克隆（warm-start）
- 目标：交叉熵让 policy 头逼近 replay 里专家的动作分布；value 头回归该局胜负 `z`。
- 作用：给后续 RL 一个像样的先验，加速收敛。**天花板 = replay 里 agent 的水平**，靠它一个人突破不了。

### 阶段② maskable PPO（可验证奖励）
- 算法：PPO + **action masking**（合法 option 上 softmax，sb3-contrib `MaskablePPO` 思路）。
- 奖励：**稀疏、可验证的胜负**（+1/-1/0），谨慎 shaping 防 reward hacking。
- 目标变了：从「人会怎么走」→「**怎么走能赢**」，于是**能超过数据天花板**。
- 用 advantage / GAE、clip 目标；value 头做 critic。

### 阶段③ AlphaZero 式 self-play + MCTS 闭环
- self-play 用 MCTS（§7）产出更强的走法分布 π 当训练标签（policy improvement operator）；
- 网络拟合 (π, z)；新网络又强化 MCTS；循环。
- 这是冲击高分的主力路线（rule-based → 搜索 → RL 的终点）。

---

## 7. 推理时：MCTS + 隐藏信息

- **搜索**：MCTS 四步（Selection 用 PUCT 结合 policy 先验 / Expansion / 用 **value 头**评估而非随机 rollout / Backpropagation），跑 N 次后选**访问次数最多**的动作。引擎接口：`search_begin / search_step / search_end / search_release`（CLAUDE.md §6.4）。
- **隐藏信息**（对手手牌/牌库/暗置）：**determinization / IS-MCTS**——用「全卡池 − 已见牌」**采样**对手未知部分，多采样各跑一次 MCTS 取平均。`manual_coin=True` 可控硬币随机以做稳健推演。
- **时间预算**：每局 10min、CPU、可能逐步有时限 → 控**搜索深度 + 采样数**，并始终留**随机合法 fallback** 防超时判负。
- **网络的角色**：只提供 policy 先验 + value 估值；不做思维链。模型再大，搜索调不动 = 负分。

---

## 9. 实现：`model/` 包（基于官方样例的改进版）

参考 `other_works/reinforcement-learning-and-mcts-sample-code.ipynb`（官方 RL+MCTS 可跑骨架）落地，并加入两项改进。**全部已在本机 Windows 引擎上跑通验证**（引擎 `cg.dll` 在本机可加载，可本地自对弈/训练——与 CLAUDE.md §3 的 macOS 限制不同）。

### 9.1 模块地图

| 文件 | 内容 | 来源 |
|---|---|---|
| `model/encoding.py` | Observation → 稀疏特征（`EmbeddingBag` 输入）；encoder/decoder 特征工程 | 移植样例 |
| `model/network.py` | 小 Transformer `MyModel(128,2,256,1,1)` + policy/value 双头 + batch 工具 | 移植样例 |
| `model/prior.py` | 从 replay 抽**真实对手卡组先验**（决卡组动作 len==60）；缓存 `data/replay_prior.json` | **改进①基础** |
| `model/mcts.py` | MCTS（PUCT、选最多访问）+ **数据驱动 determinization** | 样例 + **改进①** |
| `model/selfplay.py` | AlphaZero 式 self-play 训练循环；用项目 deck.csv；支持续训 | 移植样例 |
| `model/imitation.py` | **replay 行为克隆 SL warm-start** | **改进②** |
| `model/agent.py` | 提交契约 `agent(obs_dict)->list[int]`，永不崩溃兜底 | 新增 |

### 9.2 两项改进（相对官方样例）

**改进① 隐藏信息 determinization 数据化（`mcts.determinize` + `prior.py`）**
样例把对手看不到的牌库/手牌/奖赏**全填常量**（Snorlax `[1072]` / 基础能量 `[1]`，注释明说 no deep meaning），搜索对对手能力预测严重失真。改为：扫描 replay 开局的「选卡组」动作（恰好 60 张牌表）建**真实卡组先验**，`search_begin` 时从中采样对手未知卡（暗置 active 专门挑基础宝可梦）。无 replay 时自动退回常量填充，保证健壮。
> 当前先验：从 12 个 replay 得到 19 副真实牌表、107 种卡。更多 replay → 先验更准。

**改进② SL warm-start（`imitation.py`）**
样例从随机权重直接 self-play（早期慢）。我们先用下载的强对局 replay 做**行为克隆**（专家动作=+1、其余=-1；value=本局胜负；默认只学赢家），预热出 `sl_init.pth`，再交给 `selfplay --init-weights` 继续 RL，起点更高。

### 9.3 端到端用法

```bash
# 0) 下 replay（详见 tools/download_replays.py）
python tools/download_replays.py --latest 1 --limit 300
# 1) 建对手卡组先验（缓存）
python -m model.prior
# 2) SL warm-start：行为克隆 → model/out/sl_init.pth
python -m model.imitation --epochs 3
# 3) self-play RL（从 SL 权重续训）→ model/out/model_final.pth
python -m model.selfplay --init-weights model/out/sl_init.pth --iterations 5
# 4) 提交：sample_submission/main.py 里 `from model.agent import agent`，model/ 与权重随包打进 submission
```

### 9.4 已验证 / 关键参数（本机实测）

- 引擎对局循环、编码+网络、MCTS 整局、self-play 训练、SL 行为克隆、agent 提交封装 **均跑通**。
- MCTS 单步 ≈ **44ms @ 10 次搜索**（CPU），符合 10min/局预算。
- 默认：`SEARCH_COUNT=10`、`MyModel(128,2,256,1,1)`、AdamW lr=3e-4、HuberLoss（value δ=0.2 / policy δ=0.1）、self-play value 用 TD(λ=0.9)。
- ⚠️ 待办：**尚未做长训**（多轮 self-play）以验证改进①②对胜率的实际增益——见 §8。样例无改进时 5 轮自对弈到 76% vs random，可作对照基线。

### 9.5 训练实测与结论（2026-06-27，本机 4080 Laptop + 32 核）

**基础设施**：`model/parallel_selfplay.py` = 多进程并行 self-play（24 worker，每进程独立引擎 +
CPU 单线程推理）+ 主进程 GPU 训练。实测 collect ≈ 8s（搜索10）/ 14–23s（搜索20），train < 1s。
评估/对战用 `model/arena.py`（支持 `model:<path>` / `rule` / `random` 对手）。

**关键发现（用「对 rule-based agent 胜率」做判别尺子，打随机对手都 90%+ 无区分度）**：

| 模型 | vs rule | 说明 |
|---|---|---|
| `sl_init`（纯 SL warm-start） | ~59% | 起步即 92% vs random（样例从零是 20%）→ **改进② SL 预热价值确凿** |
| `model4`（SL + RL 4 轮，lr=3e-4） | **64%** | 头对头打 sl_init 也 57%（80–60/140 局）→ 早期 RL **确有提升** |
| `model12`（RL 12 轮） | 46% | 比 SL 还差 → **RL 跑久了自毁** |

**核心结论：当前 self-play RL 会系统性腐蚀强 SL 初始策略。**
- 第一轮 lr=3e-4：win% 序列 `92,85,85,94,94,90,60,48,69,88,88,77,81`（vs random），高方差 + 无上升趋势。
- 第二轮「修正版」（lr=3e-5、搜索20、对 rule 评估、keep-best+早停）：`53,53,35,20,27` **单调退化**，4 轮无新高早停 → 证明降 lr/加搜索**没救回**，问题在算法本身。
- 推断根因：① 样例的 **policy 标签是非标准 advantage**（非 AlphaZero 访问次数分布），浅搜索下噪声大；② **无 SL 锚**→灾难性遗忘；③ **自己打退化的自己**→负反馈螺旋；④ 我们 replay 少（仅 12 个→613 SL 样本 / 19 副先验），数据不足。

**当前最佳交付**：`model/out/model_final.pth = model4`（64% vs rule，经多局验证）。`model4_backup.pth`、`sl_init.pth` 一并保留。

**已实现的修复（`--mode az`，`parallel_selfplay.py` + `selfplay.train_on_az`）**：
1. ✅ **早停 + keep-best**（`--eval-opponent rule`、`--patience`）——保证交付不退化。
2. ✅ **标准 AlphaZero policy 标签**：MCTS **访问次数分布** + 交叉熵（`mcts_agent(policy_target="visit")`），替掉 advantage 回归；value 目标 = 最终胜负 z。
3. ✅ **SL 锚**：每轮训练混入 613 个 replay 行为克隆 one-hot 样本，防遗忘。

**AZ 实验结果（lr=1e-4、搜索20、SL锚、对 rule 评估）**：
- win% 序列 `60,70,45,63,63,45,58`（早停）——**震荡但不再单调崩**（对比 advantage 的 `53→35→20`）。**修复成功解决了灾难性自毁**。
- 但 **AZ 最佳(70% vs rule) 头对头打 model4 = 102–98 = 51%**（200 局），**统计平手**——AZ 没能显著超过 SL+早期RL 的 baseline。

**最终结论：算法修复让训练「稳定不自毁」（重要），但当前数据/搜索预算下 RL 摸不到更高天花板。瓶颈是数据与搜索深度，不是算法。**

**真正能突破的下一步（按 ROI）**：
1. **更多 replay**（最高优先）：`tools/download_replays.py` 多拉几天 → 更强 SL + 更准 prior + 更厚 SL 锚。当前仅 12 个 replay（613 SL 样本 / 19 副先验）严重受限。
2. **更深搜索**（30–50）让访问次数分布更可信（20 次太浅，分布噪声大）。
3. **冻结强对手池**（model4 + rule + 历史版本）替自对弈，信号更稳。
4. **更大评估局数**（当前 60 局 ±12% 噪声）以可靠选最佳。

## 8. 待定 / TODO

- [ ] observation→张量 的**精确特征清单与维度**（先实现 DeepSets 版并固定 schema）。
- [ ] 卡池大小 / 卡嵌入维度 / trunk 维度的**实测选型**（以 CPU 单次前向延迟为约束）。
- [ ] 多选 / 特殊状态 step 的 label 与 loss 细节。
- [ ] MCTS 的 PUCT 常数、采样数、深度、单步时限的**实测调参**。
- [ ] 对手隐藏信息的采样策略升级（当前=按真实卡组先验单次采样；可加**多 determinization 平均**、用 `logs` 做贝叶斯收窄）。
- [ ] 模型权重打包大小 & 加载延迟验证（随 `submission.tar.gz`）。
- [ ] **验证改进①②的实际增益**：长训三组对照（随机初始化 / +改进①真实 prior / +改进②SL warm-start），比 vs random 与互相对战胜率。官方样例无改进时 5 轮到 76%，作基线。
- [ ] 我方剩余牌库的 determinization 也可更准（当前从全 60 牌随机采，可扣除已见手牌/弃牌/场上）。

> 实现已落地为 `model/` 包（§9），引擎在本机 Windows 可跑；远程 Linux x86-64 同样可跑（CLAUDE.md §3）。
