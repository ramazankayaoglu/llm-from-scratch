import torch.nn as nn
from .master_mlp import MasterMLP
from .master_multi_head_attention import MasterMultiHeadAttention
from .master_layer_normalization import MasterLayerNormalization

class MasterDecoderBlock(nn.Module):
    def __init__(self, embedding_dim, num_heads, context_length):
        super().__init__()

        self.self_attention = MasterMultiHeadAttention(embedding_dim, embedding_dim, context_length, num_heads, dropout_rate = 0.5)
        self.normalization1 = MasterLayerNormalization(embedding_dim)
        self.mlp = MasterMLP(embedding_dim, embedding_dim)
        self.normalization2 = MasterLayerNormalization(embedding_dim)

    def forward(self, x):
        residual = self.normalization1(x)

        x = self.self_attention(x)
        x = self.normalization1(x)

        x = x + residual

        residual = self.normalization2(x)
        x = self.mlp(x)
        x = self.normalization2(x)

        x = x + residual

        return x

