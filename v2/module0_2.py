# the capital of france is paris 

promt = "the capital of france is"

next_token = "paris"

output = "the capital of france is paris"

"""
from transformers import AutoTokenizer, AutoModelForCausalLM, Gemma3ForCausalLM

gemma_tokenizer = AutoTokenizer.from_pretrained("google/gemma-3-1b-it")
gemma_model = AutoModelForCausalLM.from_pretrained("google/gemma-3-1b-it")
model2 = Gemma3ForCausalLM.from_pretrained("google/gemma-3-1b-it")

print(gemma_model)
print(model2)"""
"""
from tokenizer import Tokenizer

tokenizer = Tokenizer("tokenizer.json")
ids = tokenizer.encode(promt)
print(ids)
print(tokenizer.decode(ids))"""


girdi = "the letter capital of the united states is not"
cikti = "letter capital of the united is not London"

with open("text.txt", "r") as f:
    text = f.read()

#print(text)


import token
from tokenizer import Tokenizer
tokenizer = Tokenizer("tokenizer.json")


token_ids = tokenizer.encode(text)
#print(token_ids)

# save ids
ids_text = ""

for token_id in token_ids:
    ids_text += f"{token_id} "

with open("token_ids.json", "w") as f:
    f.write(ids_text)


import torch
from torch.utils.data import DataLoader
from text_dataset import TextDataset



def create_data_loader(token_ids: list, context_length: int, stride: int, batch_size: int, shuffle: bool = False, device: str = "cpu"): #batch_size fine tuning yani düzeltme yapma sayısı  
    dataset = TextDataset(token_ids, context_length, stride)                                                                           #1 dersek her adımda düzeltme yapar 5 dersek 5 adımda 1 kez düzeltme yapar 
    dataloader = DataLoader(dataset, batch_size = batch_size, shuffle = shuffle, generator = torch.Generator(device = device))                 #device nerede işlem yapsın
                                                                                                                                       #Shuffle datayı her aldığnda tekrar tekrar karıştırsın mı 
                                                                                                                                       #generator dataloader'dan next item next item diye sıradaki veriyi alır
    return dataloader
