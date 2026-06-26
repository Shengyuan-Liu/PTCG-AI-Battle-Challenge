"""PTCG AI Battle — 模型与训练代码包。

基于官方 RL+MCTS 样例（other_works/reinforcement-learning-and-mcts-sample-code.ipynb）
改进而来。改进点见 docs/model.md：
  1. 隐藏信息更真实的 determinization（数据驱动先验 + 多次采样平均）。
  2. 可选的 replay SL warm-start（行为克隆）让 self-play 起点更高。
  3. 用项目自己的 deck.csv，并提供符合提交契约的 agent 封装。

模块：
  encoding.py  —— Observation → 稀疏特征（encoder/decoder 输入）。移植自样例。
  network.py   —— 小 Transformer policy+value 网络。移植自样例。
  prior.py     —— 从 replay 抽对手卡组先验（改进 1 的数据基础）。
  mcts.py      —— MCTS + 改进的 determinization。
  selfplay.py  —— self-play 训练循环（AlphaZero 式）。
  imitation.py —— replay → 样本 + 行为克隆预训练（改进 2）。
  agent.py     —— 提交契约封装 agent(obs_dict)->list[int]。
"""
