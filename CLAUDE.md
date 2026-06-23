# CLAUDE.md — PTCG AI Battle Challenge

本文件记录该项目的背景、结构、运行方式、引擎 API 速查、策略路线图与已知问题，供后续会话快速恢复上下文。语言以中文为主，代码 / API 名称保留英文。

> 维护提示：本文件是「项目记忆」。环境/平台、版本、关键决策变化时务必更新。最近一次大更新：2026-06-23（项目从 Windows 迁到 macOS，确认引擎无法在 Mac 原生运行，决定引擎/训练走远程 Linux）。

---

## 1. 项目背景

- **比赛**：Kaggle —「The Pokémon Company - PTCG AI Battle Challenge」，主办方 = 宝可梦公司 + 东京大学松尾研究所（Matsuo Institute）+ HEROZ。2026-06-16 开赛。
  - 任务：用主办方**固定卡池**组一副 **60 张卡组**，编写 AI agent，在 `cabt` 引擎上与其他参赛者 1v1 对战。核心难点是**隐藏信息**（看不到对手手牌/牌库/暗置卡），需要前向规划与不确定性决策；官方明确说纯 rule-based 很难拿高分。
  - **比赛分两个 Category（两个独立 Kaggle 竞赛页）**，务必分清：
    1. **Simulation（模拟天梯）** = 当前项目对应的 `pokemon-tcg-ai-battle`。提交 agent，服务器 24h 不间断自动对战排名。**本身无奖金**，但是 Strategy 赛的基础。
    2. **Strategy（策略赛）** = 独立竞赛页 `...-strategy`。提交一份**书面报告**（讲策略逻辑/卡组设计），人工评审。**奖金全在这里**。
  - 链接：https://www.kaggle.com/competitions/pokemon-tcg-ai-battle
- **对战引擎**：`cabt`（Card Battle），松尾研究所提供，注册为 `kaggle_environments` 环境。文档 https://matsuoinstitute.github.io/cabt/ （api.html / game.html / sim.html / utils.html）。
- **提交形式**：打包 `submission.tar.gz`（`tar -czvf submission.tar.gz *`），顶层含 `main.py`（实现 `agent(obs_dict)->list[int]`，**不能嵌在子目录**）、`deck.csv`（60 张）、`cg/` 引擎封装（含 `libcg.so`）。Kaggle 运行时把文件放到 `/kaggle_simulations/agent/`（见 `main.py` 的路径回退）。

### 1.1 比赛规则/约束要点（据公开信息整理，**以 Kaggle 官方 Rules/Data 页为准**）

- **评分**：高斯分布 N(μ,σ²) 的技能评分（TrueSkill 类）；只看胜/负/平不看比分差；新提交约 μ₀≈600 起；**只计每队最近 2 个提交**；Simulation 每队**每天最多 5 次提交**。
- **时限**：每名玩家**每局最多 10 分钟**，用尽即判负。单步是否有秒级时限未确认。
- **算力**：**不保证有 GPU**；结合 10min/局 → 推理要快、模型要小、能 CPU 跑。运行时**是否联网/包白名单/内存/包体积上限：未在公开页确认，需登录 Kaggle Rules 页核实**。
- **允许**：RL/深度学习/self-play **明确鼓励**；外部数据/预训练模型未见禁止（但运行大概率无外网 → 权重要打进包）。卡池**固定**，只能用指定卡表；不强制 agent 自己动态选卡（可用预设卡组）。
- **关键日期（2026，公开信息，需核实）**：Simulation 组队截止 ~8/9、最终提交 ~8/16（天梯再跑约两周定榜）；Strategy 组队 ~9/6、报告截止 ~9/13；决赛在东京线下（年内稍晚）。
- **奖金（公开信息）**：第一轮 Strategy Top 8 各 **$30,000**；决赛冠 **$50,000** / 亚 **$30,000** + 决赛者 Google Cloud credits。**Simulation 无现金奖**。

---

## 2. 用户需求 / 目标 / 当前状态

- 用户在参赛，目标是有竞争力的 PTCG agent。当前 `sample_submission/main.py` 只是**随机合法走子**的占位实现，需替换为有策略的实现。
- 路线（用户定）：先 **rule-based 策略** → 再加 **搜索/前向推演** → 再上 **RL（PPO 或其它）**，reward 用 **verifiable reward（可验证的稀疏胜负）**。详见 §7。
- 用户用**中文**沟通，回复用中文。沟通风格：偏好**直接动手、果断行动**，不喜欢反复的探查式命令（少问多做、给方案）。

---

## 3. 环境与运行方式（**2026-06-23 重大变更**）

> ⚠️ 项目已从原 Windows + conda(`pokemon`, `D:\...`) 迁到 **macOS（Apple Silicon, arm64）**。原 CLAUDE.md 里的 `conda run -n pokemon` / `D:\miniconda3\...` / PowerShell **已作废**。

### 3.1 引擎的平台约束（**已实测，关键**）

- 原生库只有两个：`cg/cg.dll` = **Windows x86-64**；`cg/libcg.so` = **Linux x86-64 ELF**（BuildID `0747c564...`）。**没有任何 macOS / arm64 版本。**
- `cg/sim.py` 在非 Windows 上 `ctypes.cdll.LoadLibrary("libcg.so")`；在本机 arm64 macOS 上实测报错 `dlopen(...): slice is not valid mach-o file`（Linux ELF 不是 Mach-O，加载失败）。
- **结论：cabt 引擎无法在这台 Mac 上原生运行**——`test.py`、self-play、`search_*` 推演、RL 训练在 Mac 本地都跑不了。

### 3.2 分工（用户已选定）

- **本机 Mac**：项目内 **`.venv`**（不用 conda），只做**纯 Python**工作：卡表分析、写/读 agent 逻辑、原型。已装 `numpy`、`pandas`。
  - 建/用法：`python3 -m venv .venv` → `source .venv/bin/activate` → `pip install -r requirements.txt`。
  - 注意：本机 `python3` = 3.14.5，**装不上 `kaggle_environments`**（其依赖 `litellm` 要求 Python `<3.14`），这是预期；Mac 上也 `import` 不了 `cg.api`（加载原生库即失败）。Mac 端不依赖这些。
- **远程 Linux x86-64（跑引擎 / self-play / 训练）**：用 **Python 3.10–3.13**（推荐 3.11，与 Kaggle 对齐），`pip install -r requirements.txt` 可正常装 `kaggle_environments==1.30.1`，`libcg.so` 原生加载。**所有引擎相关运行在这里做。**
- 备选：**Windows**（用 `cg.dll`，用户也有 Windows 机）；或 **Docker `--platform linux/amd64`** 在 Mac 上模拟跑（未采用）。

### 3.3 两种驱动引擎的方式

1. **`cg.game` 直接驱动（self-play / RL 首选，无需 kaggle env）**：`battle_start(deck0, deck1)` → 循环 `battle_select(option_index_list)` → `battle_finish()`；`visualize_data()` 取可视化数据。轻量、快。
2. **`kaggle_environments` 的 `make("cabt", ...)` + `env.run([agent, agent])`（= `sample_submission/test.py`）**：跑完整对局并 `env.render(mode="html")` 出回放。
   - ⚠️ 坑：PyPI 的 `kaggle_environments` **不含 `cabt` 环境**；`make("cabt")` 需要把比赛分发的 `envs/cabt/`（`cabt.py`/`cabt.json`/visualizer）放进已安装的 `kaggle_environments` 目录。本仓库目前**没有**这些 cabt 环境文件，只有 `cg/` 封装 → 在远程 Linux 上要么补全 cabt 环境文件，要么直接用方式 1（`cg.game`）。

### 3.4 跑测试（在远程 Linux / Windows 上）

```bash
# 方式2：完整对局 + 回放（需 cabt 环境文件）
python sample_submission/test.py     # 成功打印 "Simulation finished." 并写 result.html
```
启动时若有一大段红色 `OpenSpiel exception: Unknown game 'universal_poker'/'repeated_poker' ... games skipped: 2` —— 是 `kaggle_environments` 加载 OpenSpiel 列表的告警，**与 cabt 无关，可忽略**。

---

## 4. 仓库结构

```
PTCG-AI-Battle-Challenge/            # git 仓库根（已迁到 macOS）
├── CLAUDE.md                        # 本文件（项目记忆）
├── README.md                        # 仅标题
├── requirements.txt                 # 依赖（含 Python 版本/平台注意事项）
├── .venv/                           # 本机 Mac 虚拟环境（已 gitignore）
├── cg/                              # cabt 引擎 Python 封装（核心）
│   ├── api.py     # 核心 API：所有 Enum / dataclass / to_observation_class / all_card_data / all_attack / search_*
│   ├── sim.py     # ctypes 绑定、StartData/SerialData、Battle 单例（import 时即 LoadLibrary + GameInitialize）
│   ├── game.py    # battle_start/battle_select/battle_finish/visualize_data
│   ├── utils.py   # to_dataclass / json_to_dataclass（JSON→dataclass 递归转换）
│   ├── cg.dll     # Windows x86-64 原生库（git 跟踪，未被 *.so 规则忽略）
│   └── libcg.so   # Linux x86-64 原生库（git 跟踪）
├── sample_submission/               # 提交模板（与 cg/ 同一份封装）
│   ├── main.py    # 提交入口：agent(obs_dict)->list[int]（当前=随机合法走子，待改）
│   ├── test.py    # 本地自对战 + 写 result.html
│   ├── deck.csv   # 60 张卡组（见 §9）
│   ├── result.html# 回放（renderer 缺失，见 §8）
│   └── cg/        # 同根目录 cg/（含 cg.dll + libcg.so）
├── data/
│   ├── EN_Card_Data.csv   # 全卡池静态数据 CSV（列见下），离线分析用
│   ├── meg_rulebook_en.pdf# 官方规则书（37MB；引擎规则以 cg/ 为准，二者可能不同）
│   └── data.ipynb
└── docs/venv.md                     # 旧的建环境说明（conda 版，已过时，以 §3 为准）
```

`data/EN_Card_Data.csv` 列：`Card ID, Card Name, Expansion, Collection No., Stage/Type, Rule, Category, Previous stage, HP, Type, Weakness, Resistance, Retreat, Move Name, Cost, Damage, Effect Explanation`。

> 顶层 `cg/*.py` 与 `sample_submission/cg/*.py` **内容完全一致**（diff 无差异）——改一处记得两处同步，或后续抽成单一来源。

---

## 5. Agent 接口（提交契约）

`agent(obs_dict: dict) -> list[int]`（见 `sample_submission/main.py`）：

- `obs = to_observation_class(obs_dict)` 转 dataclass。
- **初始选卡组阶段**：`obs.select is None`（且 `obs.current is None`）→ 返回**恰好 60 个 card ID**（不是下标！需合法），范例从 `deck.csv` 读。
- **常规选择阶段**：返回 `obs.select.option` 的**下标列表**，引擎校验（违反即判错）：
  - 长度在 `[minCount, maxCount]` **闭区间**（`minCount` 可为 0 → 可交空列表表示「不选/跳过」）；
  - 每个元素 `0 <= i < len(option)`；
  - **不能有重复元素**。
- **健壮性是底线**：任何分支都要返回合法动作，**绝不能崩溃/超时**（兜底：随机合法选择）。

---

## 6. 引擎 API 速查（详见 `cg/api.py`，**比赛期间 Enum/字段可能新增 → 对未知值要容错，别硬穷举**）

### 6.1 Observation 树（agent 每步收到）
- `Observation`：`select: SelectData|None`、`logs: list[Log]`、`current: State|None`、`search_begin_input: str|None`（agent 收到的 obs 里非 None，传给 `search_begin`）。
- `State`：`turn`(1=先手T1,2=后手T1,3=先手T2…)、`turnActionCount`、`yourIndex`、`firstPlayer`、`supporterPlayed`/`stadiumPlayed`/`energyAttached`(手动贴能每回合1次)/`retreated`、`result`(-1未结束)、`stadium`、`looking`、`players: list[PlayerState]`(长度2)。
- `PlayerState`：`active: list[Pokemon|None]`(暗置=None)、`bench`、`benchMax`、`deckCount`、`discard`、`prize: list[Card|None]`(首=底,末=顶,暗置=None)、`handCount`、`hand: list[Card]|None`(**对手手牌=None**)、`poisoned/burned/asleep/paralyzed/confused`(出战位状态)。
- `Pokemon`：`id, serial, hp, maxHp, appearThisTurn`(刚上场→不能攻击/进化)、`energies: list[EnergyType]`(等效类型)、`energyCards`、`tools`、`preEvolution`。
- `Card`：`id, serial, playerIndex`。

### 6.2 关键 Enum（值→义）
- **SelectType**：0 MAIN / 1 CARD / 2 ATTACHED_CARD / 3 CARD_OR_ATTACHED_CARD / 4 ENERGY / 5 SKILL / 6 ATTACK / 7 EVOLVE / 8 COUNT / 9 YES_NO / 10 SPECIAL_CONDITION。
- **OptionType**（决定该 option 带哪些字段，用于读懂选项语义）：0 NUMBER(`number`) / 1 YES / 2 NO / 3 CARD(`area,index,playerIndex`) / 4 TOOL_CARD(+`toolIndex`) / 5 ENERGY_CARD(+`energyIndex`) / 6 ENERGY(+`energyIndex,count`) / 7 PLAY(`index`手牌下标) / 8 ATTACH(`area,index`+`inPlayArea,inPlayIndex`) / 9 EVOLVE(同 ATTACH 结构) / 10 ABILITY(`area,index`) / 11 DISCARD / 12 RETREAT / 13 ATTACK(`attackId`) / 14 END / 15 SKILL(`cardId,serial`；`cardId==0`=处理特殊状态) / 16 SPECIAL_CONDITION(`specialConditionType`)。
- **SelectContext**（0–48，"为什么要选"，常用）：0 MAIN / 1 SETUP_ACTIVE_POKEMON / 2 SETUP_BENCH_POKEMON / 3 SWITCH / 8 DISCARD / 18-19 EVOLVES_FROM/TO / 35 ATTACK / 37 EVOLVE / 41 IS_FIRST(选先后手) / 42 MULLIGAN(重抽) / 46 COIN_HEAD(选硬币正面) / 47-48 施加/恢复特殊状态…（全表见 `api.py:68`）。
- **AreaType**：1 DECK/2 HAND/3 DISCARD/4 ACTIVE/5 BENCH/6 PRIZE/7 STADIUM/8 ENERGY/9 TOOL/10 PRE_EVOLUTION/11 PLAYER/12 LOOKING。
- **EnergyType**：0 COLORLESS(招式费里=任意1点)/1 GRASS/2 FIRE/3 WATER/4 LIGHTNING/5 PSYCHIC/6 FIGHTING/7 DARKNESS/8 METAL/9 DRAGON/10 RAINBOW(任意)/11 TEAM_ROCKET(算超能+恶)。
- **CardType**：0 POKEMON/1 ITEM/2 TOOL/3 SUPPORTER/4 STADIUM/5 BASIC_ENERGY/6 SPECIAL_ENERGY。
- **SpecialConditionType**：0 POISON/1 BURN/2 SLEEP/3 PARALYZE/4 CONFUSE。
- **LogType**（读历史/对手行为，0–23）：含 DRAW/DRAW_REVERSE(对手抽,看不到)/MOVE_CARD(_REVERSE)/PLAY/ATTACH/EVOLVE/ATTACK(`attackId`)/HP_CHANGE/COIN(`head`)/**RESULT** 等。

### 6.3 SelectData 辅助字段
`type`+`context` 判断当前在选什么；`minCount`/`maxCount`；`remainDamageCounter`；`remainEnergyCost`(贴/付能量时还差几点)；`deck`(从牌库选时给出牌库卡数组，否则 None)；`contextCard`；`effect`(触发当前选择的卡)。

### 6.4 搜索 / 前向推演 API（做强 agent 的核心，**仅远程 Linux 可跑**）
- `search_begin(agent_observation, your_deck, your_prize, opponent_deck, opponent_prize, opponent_hand, opponent_active, manual_coin=False) -> SearchState`
  - `agent_observation` 必须**原样**是 agent 收到的 obs（`search_begin_input != None`）。各 `*_deck/prize/hand` 是对隐藏信息的**预测**，张数须 ≥ 真实数量；对手暗置 Active 需在 `opponent_active` 预测（明示则忽略该参数）；若 `select.deck != None` 则 `your_deck` 被忽略。`manual_coin=True` → 硬币正反可由你选（便于稳/最优推演）。
- `search_step(search_id, select) -> SearchState`：在某节点施加一次选择（同 §5 契约），返回新 `SearchState`（含 `observation` 与 `searchId`）。保存 `searchId` 用不同 `select` 多次 step 即可**分叉枚举走法**。
- `search_end()`：结束搜索（内存池化复用）。`search_release(search_id)`：释放某分支内存。
- 隐藏信息 → 用「全卡池 − 已见牌」采样对手未知部分，多采样跑多次取平均（determinized / IS-MCTS 思路）。注意 10min/局 → 控深度+采样数+留兜底。

### 6.5 静态数据
- `all_card_data() -> list[CardData]`：`cardId,name,cardType,retreatCost,hp,weakness,resistance,energyType,basic/stage1/stage2,ex`(被击倒送对手2张prize)`,megaEx`(送3张)`,tera`(在备战不受招式伤害)`,aceSpec`(一套牌≤1张)`,evolvesFrom,skills:list[Skill](name/text),attacks:list[int]`(attackId)。
- `all_attack() -> list[Attack]`：`attackId,name,text,damage,energies:list[EnergyType]`。
- 建议 agent 启动时把两者各建 `{id: data}` 字典缓存。

### 6.6 胜负判定（`LogType.RESULT`，引擎写死，务必记牢）
`result`：0=玩家0胜/1=玩家1胜/2=平。`reason`：**1=拿光6张奖赏 / 2=回合开始时牌库为0(deck-out判负) / 3=出战位无宝可梦 / 4=卡牌效果**。reward：胜+1/负-1/平0（`cabt.json`）。

---

## 7. 策略路线图（rule-based → 搜索 → RL）

- **阶段0 脚手架（Mac 可写代码，远程 Linux 跑）**：写不依赖 kaggle env 的本地对战 runner（用 `cg.game` 直驱）+ 胜率统计 + observation 解析/估值工具 + 卡表缓存。
- **阶段1 Rule-based baseline**：底线=永不崩溃+永远合法（兜底随机）。启发式优先级：布场选高 HP 主攻手→进化→贴能量给主攻手→Supporter 过牌→Item→攻击（能一击击倒就打，算弱点/抗性，警惕 ex/megaEx 送 2/3 prize）→濒死按 retreatCost 撤退；处理特殊状态/先后手/mulligan。目标：稳定 baseline（μ₀≈600 起）。
- **阶段2 搜索增强**：`search_begin/step` 回合内 1~N 步 lookahead + 估值函数（prize 差、场上总 HP、能量布署、手牌质量）；对手隐藏信息多采样平均；`manual_coin=True` 控随机。
- **阶段3 RL（PPO + verifiable reward）**：observation→定长向量；变长 option list→**action masking**（仅合法选项 softmax，maskable PPO）；reward 以可验证稀疏胜负为主，谨慎 shaping 防 reward hacking；self-play 对手池=历史版本+rule-based。**无 GPU+10min/局 → 模型小、CPU 推理快、权重打进包**。
- **阶段4 迭代**：本地大量自对战算胜率（但**天梯才是最终裁判**，每天5次/最近2个计分）；固定卡池下迭代 60 张；**最后写 Strategy 报告（拿奖金的关键）**。

---

## 8. 已知问题

- **`result.html` 回放无法显示（renderer 缺失，真实问题）**：末尾 `window.kaggle.renderer = ;`（等号右边空）。根因：cabt 的 `html_renderer()` 找 `visualizer/default/dist/index.html`→`cabt.js` 都不存在，返回空串。**对局数据本身完整合法**（内嵌 `window.kaggle={...}` 是有效 JSON），只缺前端播放器 JS。修复：从官方分发补 visualizer 资源；或用社区 `cabt-viewer`；只看结果可不管。
- **引擎在 Mac 不可运行**（见 §3.1）。
- **PyPI `kaggle_environments` 不含 cabt 环境**（见 §3.3 坑）。

---

## 9. 约定 / 注意事项

- **引擎相关一律在远程 Linux x86-64（或 Windows）跑**；Mac 本地只做纯 Python（`.venv`，`numpy/pandas`）。远程用 Python 3.10–3.13。
- 原生引擎是**单例**：`Battle.battle_ptr`（对局）/ `agent_ptr`（搜索，`cg.api` 模块级全局）全局共享；一局结束需 `battle_finish()` 释放。
- 提交包必须含 `libcg.so`（已确认被 git 跟踪、未被 `.gitignore` 的 `*.so` 忽略）；`main.py` 须在 tar 顶层。
- card ID `3` = Basic {W} Energy。当前 `sample_submission/deck.csv` = 35×3(水能量) + 4×{723,722,1235,1227,1145} + 2×{721,1205} + 1×1158，共 60。
- 写 agent 对未知 Enum 值/缺失字段要**防御性容错**（官方注释多处提示比赛期间会新增）。
