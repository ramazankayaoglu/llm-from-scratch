from json import decoder
import torch
from .master_model import MasterModel
from .master_tokenizer import MasterTokenizer
import master_tokenizer

master_tokenizer = MasterTokenizer("tokenizer.json")

prompt = "the capital of united"

tokens = master_tokenizer.encode(prompt)

#print(tokens.shape) 

torch.manual_seed(1)    


master_model = MasterModel(vocab_size=len(master_tokenizer.vocab), embedding_dim= 4, num_heads=4, context_length=32)

sentence_meanings_with_attention_context = master_model(tokens)
#print(sentence_meanings_with_attention_context)

from master_layer_normalization import MasterLayerNormalization


out = master_model(tokens)
#print(out)

norm_layer = MasterLayerNormalization(4)
#print(norm_layer(out))


from transformers import AutoTokenizer, AutoModelForCausalLM

#q_tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B")
#q_model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-0.6B")







    