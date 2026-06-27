"""小 Transformer encoder-decoder：policy + value 双头网络。

忠实移植自官方样例。结构：
  encoder: EmbeddingBag(sum) → TransformerEncoder → value 头 tanh(mean)
  decoder: 候选动作 EmbeddingBag(sum) → 对 encoder 输出做 cross-attention → 每动作一个 policy 标量
默认规模 MyModel(128, 2, 256, 1, 1)，极小、CPU 友好（MCTS 要高频调用）。
"""
import torch
import torch.nn
import torch.nn.functional

from .encoding import (
    SparseVector, decoder_size, encoder_size, num_words_encoder,
)


class DecoderLayer(torch.nn.Module):
    def __init__(self, d_model: int, num_heads: int, d_feedforward: int):
        super().__init__()
        self.attention = torch.nn.MultiheadAttention(d_model, num_heads)
        self.fc1 = torch.nn.Linear(d_model, d_feedforward)
        self.fc2 = torch.nn.Linear(d_feedforward, d_model)
        self.norm1 = torch.nn.LayerNorm(d_model)
        self.norm2 = torch.nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor, encoder_out: torch.Tensor) -> torch.Tensor:
        y, _ = self.attention(x, encoder_out, encoder_out, need_weights=False)
        res = self.norm1(x + y)
        y = self.fc1(res)
        y = torch.nn.functional.relu(y)
        y = self.fc2(y)
        return self.norm2(res + y)


class MyModel(torch.nn.Module):
    def __init__(self, d_model: int, num_heads: int, d_feedforward: int,
                 num_layers_encoder: int, num_layers_decoder: int):
        super().__init__()
        self.d_model = d_model
        self.encoder_bag = torch.nn.EmbeddingBag(encoder_size, d_model, mode="sum")
        encoder_layer = torch.nn.TransformerEncoderLayer(d_model, num_heads, d_feedforward, 0)
        self.encoder = torch.nn.TransformerEncoder(encoder_layer, num_layers_encoder, enable_nested_tensor=False)
        self.encoder_fc = torch.nn.Linear(d_model, 1)
        self.decoder_bag = torch.nn.EmbeddingBag(decoder_size, d_model, mode="sum")
        self.decoder = torch.nn.ModuleList(
            DecoderLayer(d_model, num_heads, d_feedforward) for _ in range(num_layers_decoder)
        )
        self.decoder_fc = torch.nn.Linear(d_model, 1)

    def forward(self, index_encoder, value_encoder, offset_encoder,
                index_decoder, value_decoder, offset_decoder):
        v = self.encoder_bag(index_encoder, offset_encoder, value_encoder)
        v = v.reshape(-1, num_words_encoder, self.d_model).transpose(0, 1)
        batch_size = v.size(1)
        encoder_out = self.encoder(v)
        v = self.encoder_fc(encoder_out)
        v = torch.tanh(v.mean(0))

        p = self.decoder_bag(index_decoder, offset_decoder, value_decoder)
        p = p.reshape(batch_size, -1, self.d_model).transpose(0, 1)
        for layer in self.decoder:
            p = layer(p, encoder_out)
        p = self.decoder_fc(p)
        p = p.transpose(0, 1).view(batch_size, -1)
        p = torch.tanh(p)
        return v, p


def eval_nn(sv_enc: SparseVector, sv_dec: SparseVector, model: MyModel):
    """单局面评估：返回 (value: float, policy: list[float])。"""
    device = next(model.parameters()).device
    value, policy = model(
        torch.tensor(sv_enc.index, dtype=torch.int32, device=device),
        torch.tensor(sv_enc.value, dtype=torch.float32, device=device),
        torch.tensor(sv_enc.offset, dtype=torch.int32, device=device),
        torch.tensor(sv_dec.index, dtype=torch.int32, device=device),
        torch.tensor(sv_dec.value, dtype=torch.float32, device=device),
        torch.tensor(sv_dec.offset, dtype=torch.int32, device=device))
    return value.tolist()[0][0], policy.tolist()[0]


# 网络架构（训练与 agent 推理共用此处，保证一致）。
# 旧版 v4 = (128, 2, 256, 1, 1)。纯 NN 单次前向后无搜索延迟压力 → 放大。
# 可用环境变量覆盖（便于扫超参，不改码）：PTCG_DMODEL / PTCG_HEADS / PTCG_FF / PTCG_ENC / PTCG_DEC
import os as _os
ARCH = dict(
    d_model=int(_os.getenv("PTCG_DMODEL", 256)),
    num_heads=int(_os.getenv("PTCG_HEADS", 4)),
    d_feedforward=int(_os.getenv("PTCG_FF", 1024)),
    num_layers_encoder=int(_os.getenv("PTCG_ENC", 3)),
    num_layers_decoder=int(_os.getenv("PTCG_DEC", 3)),
)


def new_model(device=None, **override) -> MyModel:
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg = {**ARCH, **override}
    return MyModel(cfg["d_model"], cfg["num_heads"], cfg["d_feedforward"],
                   cfg["num_layers_encoder"], cfg["num_layers_decoder"]).to(device)


class LearnInput:
    """把多个 SparseVector 拼成一个 batch 的 EmbeddingBag 输入。"""
    def __init__(self):
        self.index: list[int] = []
        self.value: list[float] = []
        self.offset: list[int] = []

    def add(self, sv: SparseVector):
        count = len(self.index)
        self.index.extend(sv.index)
        self.value.extend(sv.value)
        for o in sv.offset:
            self.offset.append(o + count)
