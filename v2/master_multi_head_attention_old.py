import torch
import torch.nn as nn

from master_causal_attention import MasterCausalAttention

class MasterMultiHeadAttention(nn.Module): 
    def __init__(self, embedding_dim, output_dim, context_length, num_heads, dropout_rate = 0):
        super().__init__()
        
        self.heads = nn.ModuleList([MasterCausalAttention(embedding_dim, output_dim, context_length, dropout_rate) for _ in range(num_heads)])

        self.projection = nn.Linear(embedding_dim, output_dim)

    def forward(self, x):
      attention_outputs = []
      for head in self.heads:
        head_output = head(x)
        attention_outputs.append(head_output)

      return torch.cat(attention_outputs, dim = 1)
        