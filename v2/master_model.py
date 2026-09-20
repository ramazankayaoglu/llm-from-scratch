from turtle import forward
import torch
import torch.nn as nn

from master_decoder_block import MasterDecoderBlock
from master_embedding import MasterEmbedding

class MasterModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim, num_heads, context_length, num_layers, device):
        super().__init__()
        self.embedding = MasterEmbedding(vocab_size, embedding_dim, context_length, device)
        self.layers = nn.Sequential(*[MasterDecoderBlock(embedding_dim, num_heads, context_length, device) for _ in range(num_layers)])

        self.lm_head = nn.Linear(embedding_dim, vocab_size, device = device)
        self.device = device

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

    def generate(self,
                 x:
                torch.Tensor,
                max_new_tokens: int,
                temperature : float = 1.0): # top_k, top_p, temperature
        tokens = x.detach().cpu().numpy().tolist()

        for _ in range(max_new_tokens):
            x = x.unsqueeze(0).to(self.device)
            out = self.forward(x)
            out = out.squeeze(0)    
            if temperature != 1.0 and temperature > 0:
                adjusted_outs = out[-1] / temperature
                probs = torch.softmax(adjusted_outs, dim = -1)
            else:
                probs = torch.softmax(out[-1], dim=-1)
            #_, max_index = torch.max(probs, dim=-1)
            sample = torch.multinomial(probs, 1)
            tokens.append(sample.item())
            if sample == 59 or len(tokens) > 32: #<eos> and max context length
                break

            x = torch.tensor(tokens)
        return tokens