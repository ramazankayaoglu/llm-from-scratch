from .master_causal_attention import MasterCausalAttention
from .master_model import MasterModel
from .master_tokenizer import MasterTokenizer
import torch

master_tokenizer = MasterTokenizer("tokenizer.json")


prompt = "the capital of united"
tokens = master_tokenizer.encode(prompt).long()
master_model = MasterModel(
    vocab_size=len(master_tokenizer.vocab),
    embedding_dim=4,
    context_length=32
)
sentence_meanings_with_attention_context = master_model(tokens)
#print(sentence_meanings_with_attention_context)

q_weights = torch.nn.Linear(4,3, bias = False)
k_weights = torch.nn.Linear(4,3, bias = False)
v_weights = torch.nn.Linear(4,3, bias = False)


q_of_sentence = q_weights(sentence_meanings_with_attention_context)
k_of_sentence = k_weights(sentence_meanings_with_attention_context)
v_of_sentence = v_weights(sentence_meanings_with_attention_context)


attention_scores = q_of_sentence @ k_of_sentence.T
attention_weights = torch.softmax(attention_scores / k_of_sentence.shape[-1] ** 0.5, dim = 1)

mask = torch.tril(torch.ones(attention_weights.shape[0], attention_weights.shape[0]))
#print(mask)
#print(attention_weights * mask)

masked_attention_weights = attention_weights.masked_fill(mask == 0, -torch.inf)
#print(masked_attention_weights)
softmax_masked_self_attention = torch.softmax(masked_attention_weights, dim = 1)

dropout_rate = 0.5
torch.manual_seed(1) 
dropout = torch.nn.Dropout(dropout_rate)

#print(dropout(softmax_masked_self_attention))


context_vector = attention_weights @ v_of_sentence

#print(context_vector)


from plot_tokens import plot_tokens

q_k_v_sentences = [
    {
    "words" : q_of_sentence.detach().numpy(), #sözlükteki yeri
    "labels": master_tokenizer.tokenize(prompt),
    "color": "blue"
    },
    {
    "words" : k_of_sentence.detach().numpy(), #sıra bilgisi de eklendi
    "labels": master_tokenizer.tokenize(prompt),
    "color": "purple"
    },    
    {
    "words" : v_of_sentence.detach().numpy(), #token bu cümlenin içinde ne kadar önemli ve anlam da eklendi 
    "labels": master_tokenizer.tokenize(prompt),
    "color": "orange"
    },
      {
    "words" : context_vector.detach().numpy(), #token bu cümlenin içinde ne kadar önemli ve anlam da eklendi 
    "labels": master_tokenizer.tokenize(prompt),
    "color": "green"
    }
]

#plot_tokens(q_k_v_sentences, "Query Key Value Sentence Space")




#print(q_weights.weight) 
#print(q_weights(sentence_meanings))



#Causal Self Attention (Nedensel Self Attention)



#Multi-Head Attention
import torch
import torch.nn as nn

class MasterMultiHeadAttention(nn.Module):
    def __init__(self, embedding_dim, output_dim, context_length, num_heads, dropout_rate = 0):
        super().__init__()
        
        self.heads = nn.ModuleList([MasterCausalAttention(embedding_dim, output_dim, context_length, dropout_rate) for _ in range(num_heads)])

    def forward(self, x):
      attention_outputs = []
      for head in self.heads:
        head_output = head(x)
        attention_outputs.append(head_output)

      return torch.cat(attention_outputs, dim = 1)


multi_head_attention = MasterMultiHeadAttention(embedding_dim=4, output_dim=4, context_length=32, num_heads=2, dropout_rate=0)

out = multi_head_attention(torch.randn(4,4))

print(out.shape, out)
