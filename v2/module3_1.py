from json import decoder
import torch
from .master_model import MasterModel
from .master_tokenizer import MasterTokenizer
import master_tokenizer

master_tokenizer = MasterTokenizer("tokenizer.json")

prompt = "the capital of united"

tokens = master_tokenizer.encode(prompt)

torch.manual_seed(1)    


master_model = MasterModel(vocab_size=len(master_tokenizer.vocab), embedding_dim= 4, num_heads=4, context_length=32, num_layers = 3)

out = master_model(tokens)
#print(out)
#print(out.shape)

#print(master_model)
#print(out[0])

import torch

#probs = torch.softmax(out[-1], dim = -1)
#print(probs)

#max_prob, max_index = torch.max(probs, dim = -1)
#print(max_prob, max_index)


from transformers import AutoTokenizer, AutoModelForCausalLM

q_tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B")
q_model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-0.6B")

q_tokens = q_tokenizer.encode(prompt)

#print(q_tokens)

q_tokens2 = q_tokenizer.tokenize(prompt)
#print(q_tokens2)

q_out = q_model(torch.tensor([q_tokens]))


#print(q_out.logits.shape)

last_token = q_out.logits[0, 3, :]

probs = torch.softmax(last_token, dim = -1)
max_prob, max_index = torch.max(probs, dim = -1)

#print(max_prob, max_index)
#print(q_tokenizer.decode(max_index))


#print(q_tokens)
#input = [1782, 6722, 315, 28192] "the capital of united"
#output = [38297, 315, 279, 5302] "instructions of the states"
#expected = [6722, 315, 28192, 5302] "capital of united states"



g_tokenizer = AutoTokenizer.from_pretrained("google/gemma-3-1b-pt")
g_model = AutoModelForCausalLM.from_pretrained("google/gemma-3-1b-pt")


g_tokens = g_tokenizer.encode("the capital of united")
#print(g_tokens)
#input = [2, 1437, 5279, 529, 26974] the capital of united
#expected = [1437, 5279, 529, 26974, 5022] the capital of united states
#output = [184, 236743, 529,  506, 5022 ] <h1>  of the states

g_out = g_model(torch.tensor([g_tokens]))
#print(g_out.logits.shape)


probs_gemma = torch.softmax(g_out.logits[0, 4, :], dim = -1)
max_prob_gemma, max_index_gemma = torch.max(probs_gemma, dim = -1)

#print(max_prob_gemma, max_index_gemma, probs_gemma)
#print(g_tokenizer.decode([[1437, 5279, 529, 26974, 5022 ]]))