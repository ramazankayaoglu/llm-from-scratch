from turtle import forward
import torch
import torch.nn as nn

from .master_decoder_block import MasterDecoderBlock
from .master_embedding import MasterEmbedding

class MasterModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim, num_heads, context_length, num_layers):
        super().__init__()
        self.embedding = MasterEmbedding(vocab_size, embedding_dim, context_length)
        self.layers = nn.Sequential(*[MasterDecoderBlock(embedding_dim, num_heads, context_length) for _ in range(num_layers)])

        self.lm_head = nn.Linear(embedding_dim, vocab_size)

    def forward(self, x):
        x = self.embedding(x) #dictionary meanings of the tokens(words)
        
        x = self.layers(x)
        x = self.lm_head(x)

        return x

    """
    out = master_model(tokens)
    probs = torch.softmax (out[-1], dim = -1)
    max_prob, max_index = torch.max(probs, dim = -1)
    #print(max_prob, max_index, probs)
    """

    def generate(self, x: torch.Tensor, max_new_tokens: int): # top_k, top_p, temperature
        tokens = x.detach().cpu().numpy().tolist()

        for _ in range(max_new_tokens):
            out = self.forward(x)
            probs = torch.softmax(out[-1], dim=-1)
            _, max_index = torch.max(probs, dim=-1)
            tokens.append(max_index.item())
            if max_index == 59 or len(tokens) > 32: #<eos> and max context length
                break

            x = torch.tensor(tokens)
        return tokens