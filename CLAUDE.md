# CLAUDE.md — PTCG AI Battle Challenge

本文件记录该项目的背景、结构、运行方式、已知问题，以及用户的需求，供后续会话快速恢复上下文。语言以中文为主，代码 / API 名称保留英文。

## 1. 项目背景

- **比赛**：Kaggle —「The Pokémon Company - PTCG AI Battle Challenge / Simulation」
  - 链接：https://www.kaggle.com/competitions/pokemon-tcg-ai-battle
  - 任务：为「宝可梦集换式卡牌游戏（Pokémon TCG）」编写一个 AI 对战 agent，与其他参赛者的 agent 进行 1v1 对战。
- **对战引擎**：`cabt`（Card Battle），由 Matsuo Institute 提供，作为一个 `kaggle_environments` 环境注册。
  - 引擎 API 文档：https://matsuoinstitute.github.io/cabt/api.html
- **提交形式**：提交一个 `main.py`，其中实现 `agent(obs_dict) -> list[int]` 函数；连同 `deck.csv`（60 张卡的卡组）和 `cg/` 引擎封装一起打包。Kaggle 运行时会把文件放到 `/kaggle_simulations/agent/` 下（见 `main.py` 里对该路径的回退处理）。

## 2. 用户需求 / 目标

- 用户正在参赛，目标是开发出有竞争力的 PTCG 对战 agent（当前 `main.py` 只是随机走子的占位实现，需要替换为有策略的实现）。
- 运行 / 调试时**务必使用 `pokemon` 这个 conda 环境**（见下方运行方式）。
- 用户用中文沟通，回复请用中文。

## 3. 环境与运行方式

- **conda 环境**：`pokemon`（路径 `D:\miniconda3\envs\pokemon`）。`kaggle_environments` 安装在
  `D:\miniconda3\envs\pokemon\Lib\site-packages\kaggle_environments\`，`cabt` 环境在其
  `envs\cabt\` 子目录。
- 本机为 **Windows**，shell 为 PowerShell。`conda` 不在 Bash 工具的 PATH 上，需用 PowerShell 调用。
- **跑本地对战测试**：
  ```powershell
  conda run -n pokemon python test.py
  ```
  成功时输出 `Simulation finished.` 并生成 `result.html`（对局回放）。
- **关于启动时的红色报错（可忽略）**：运行时会打印一大段
  `OpenSpiel exception: Unknown game 'universal_poker' / 'repeated_poker' ...`。
  这是 `kaggle_environments` 加载 OpenSpiel 游戏列表时跳过两个不可用扑克游戏产生的告警
  （日志含 `OpenSpiel games skipped: 2`），**与 `cabt` 无关，不影响运行**。PowerShell 会把子进程
  stderr 包装成 `NativeCommandError` 显示成红色，看起来吓人但只是告警。

## 4. 仓库结构（sample_submission/）

- `test.py` — 本地跑一局自对战（agent vs agent）并写出 `result.html`。
- `main.py` — **提交入口**。实现 `agent(obs_dict) -> list[int]`；当前为随机实现，待改进。
  - `read_deck_csv()` 读 `deck.csv`，返回 60 个 card ID。
  - 初始选择阶段 `obs.select == None`，此时必须返回 60 张卡的卡组。
- `deck.csv` — 60 行，每行一个 card ID 的卡组。
- `cg/` — cabt 引擎的 Python 封装（ctypes 绑定到 `cg.dll` / `libcg.so`）：
  - `cg/api.py` — **核心 API**：所有 Enum、`Observation` / `SelectData` / `Option` / `State` /
    `CardData` / `Attack` 等 dataclass，以及 `to_observation_class()`、`all_card_data()`、
    `all_attack()`、`search_begin()` / `search_step()` / `search_end()` / `search_release()`。
  - `cg/sim.py` — 加载动态库、定义 ctypes 结构与函数签名、`Battle` 类。
  - `cg/game.py` — `battle_start()` / `battle_select()` / `battle_finish()` / `visualize_data()`。
  - `cg/cg.dll`、`cg/libcg.so` — 原生引擎二进制（Win / Linux）。
- `result.html` — `env.render(mode="html")` 生成的对局回放（约 1.4MB，内嵌完整对局 JSON）。
- 父目录 `D:\Project\PTCG-AI-Battle-Challenge\` 还有 `data/`、`docs/`、`cg/`、`data.ipynb` 等（git 仓库根在父目录）。

## 5. Agent 接口（要点）

`agent(obs_dict: dict) -> list[int]`：

- 用 `obs = to_observation_class(obs_dict)` 转成 dataclass。
- `obs.select is None` → 初始选卡组阶段，返回 60 个 card ID（见 `read_deck_csv`）。
- 否则返回一个 **option 下标列表**，约束：
  - 每个元素 `0 <= i < len(obs.select.option)`；
  - 列表长度在 `[obs.select.minCount, obs.select.maxCount]`（闭区间）内；
  - **不能有重复元素**。
- `obs.select.type`（`SelectType`）+ `obs.select.context`（`SelectContext`）共同决定当前在选什么
  （出牌 / 附能量 / 进化 / 攻击 / 选数量 / Yes-No / 特殊状态等）。各 `Option.type`（`OptionType`）
  决定该选项携带哪些字段（如 `attackId`、`area`、`index`、`playerIndex` 等）。详见 `cg/api.py` 注释。

### 搜索 / 模拟 API（用于做前向推演的强 agent）

`cg/api.py` 提供基于引擎的搜索接口，可在 agent 内部推演后续局面：

- `search_begin(agent_observation, your_deck, your_prize, opponent_deck, opponent_prize, opponent_hand, opponent_active, manual_coin=False) -> SearchState`
  - `agent_observation` 必须原样传入 agent 收到的 observation（其 `search_begin_input` 不能为 None）。
  - 各 `*_deck/prize/hand` 需提供与真实数量一致的**预测** card ID；对手暗置的 Active 需预测。
- `search_step(search_id, select) -> SearchState` — 在搜索树里推进一步选择。
- `search_end()` — 结束搜索，内存留作下次复用。
- `search_release(search_id)` — 释放指定搜索状态。
- `all_card_data() -> list[CardData]`、`all_attack() -> list[Attack]` — 取全部卡牌 / 招式静态数据。

> 注意：`cg/api.py` 中多处注释提示「比赛期间 Enum / 字段可能新增」，写代码时对未知枚举值要容错。

## 6. 已知问题

### 6.1 `result.html` 可视化无法显示（renderer 缺失）— 真实存在的问题

- 现象：在编辑器 / 浏览器里，`result.html` 报 JavaScript 语法错误。
  真正出错的是末尾的 `window.kaggle.renderer = ;`（约第 142075 行，等号右边为空 → `Expression expected`）。
  （编辑器有时把错误位置错标到几万行处，那是 JS 语言服务解析超大内嵌脚本时位置串位所致。）
- 根因：`cabt` 环境的 `html_renderer()`（`envs/cabt/cabt.py`）依次找
  `visualizer/default/dist/index.html` → `cabt.js`，**两个文件在当前安装里都不存在**，于是返回空字符串，
  被填进模板就成了非法的 `window.kaggle.renderer = ;`。
- 重要：**对局数据本身完整且合法**（内嵌的 `window.kaggle = {...}` 是有效 JSON，包含每一步状态 /
  动作 / reward），模拟逻辑、胜负判定都正常。**缺的只是前端可视化播放器的 JS 资源**。
- 修复方向（待办）：从比赛官方分发获取 `cabt` 的 visualizer 资源
  （`visualizer/default/dist/index.html` 或 `cabt.js`），放到上述安装目录；或重装 / 更新完整版 `cabt` 包。
  若只关心对局结果而非回放画面，则可暂不处理。

## 7. 约定 / 注意事项

- 运行任何 Python 一律走 `conda run -n pokemon ...`（PowerShell）。
- 原生引擎是单例：`Battle.battle_ptr` 全局共享；一局结束需 `battle_finish()` 释放。
- reward 规则：胜 `+1` / 负 `-1` / 平 `0`（见 `cabt.json`）。
- card ID `3` = Basic {W} Energy（当前 deck.csv 里大量出现）。
