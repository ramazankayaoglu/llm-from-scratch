import torch
from .master_model import MasterModel
from .master_tokenizer import MasterTokenizer


#master_tokenizer = MasterTokenizer("mytokenizer.json")
master_tokenizer = MasterTokenizer("tokenizer.json")

prompt = "the capital of the united"

tokens = master_tokenizer.encode(prompt)
#print(tokens)
context_length = 32

torch.manual_seed(1)
master_model = MasterModel(vocab_size=len(master_tokenizer.vocab), embedding_dim= 12, num_heads=4, context_length=32, num_layers = 8)

out = master_model(tokens)
#print(out.shape)

with open("v1/text.txt", "r") as f:
    text = f.read()

#print(len(text), text[:100])

token_ids = master_tokenizer.encode(text)
#print(len(token_ids))

ids = token_ids.detach().cpu().numpy().tolist()
#print(len(ids), type(ids))


#from module1_1 import train_data_loader
from text_dataset import TextDataset

stride = 12 
dataset = TextDataset(ids, context_length, stride)

#print(len(dataset.inputs), len(dataset.targets))

parameters_count = sum(p.numel() for p in master_model.parameters())
#print(parameters_count)
#print(master_model)

#print(dataset.inputs[0], dataset.targets[0])

out0 = master_model(dataset.inputs[0])
#print(out0)
#print(out0.shape)

import torch.nn as nn

loss_fn = nn.CrossEntropyLoss()
loss = loss_fn(out0, dataset.targets[0])
#print(loss)

# optimizer = torch.optim.SDG(model.parameters(), lr = 1e-3) 
optimizer = torch.optim.AdamW(master_model.parameters(), lr = 1e-3)


epoch = 500
for epoch in range(epoch):
    total_loss = 0 
    for input, target in dataset:
        pred = master_model(input)
        
        loss = loss_fn(pred, target)
        total_loss += loss.item()
        
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
    
    average_loss = total_loss / len(dataset)
    print(f"Epoch {epoch + 1} loss: {loss.item()} average_loss: {average_loss}")



import torch

torch.save(master_model.state_dict(), "master_model.pth")

"""

out = master_model(tokens)
probs = torch.softmax (out[-1], dim = -1)
max_prob, max_index = torch.max(probs, dim = -1)
#print(max_prob, max_index, probs)


new_tokens = tokens.detach().cpu().numpy().tolist()
#print(new_tokens)
new_tokens.append(61)
#print(new_tokens)

out_new = master_model(torch.tensor(new_tokens))
probs_new = torch.softmax (out[-1], dim = -1)
max_prob_new, max_index_new = torch.max(probs_new, dim = -1)
#print(max_prob_new, max_index_new, probs_new)


new_tokens_2 = master_tokenizer.encode("the capital of the united states is not london. the capital of france is paris ")
new_tokens_2 = new_tokens_2.detach().cpu().numpy().tolist()
new_tokens_2.append(61)
#print(len(new_tokens_2))

"""

loaded_master_model = MasterModel(vocab_size=len(master_tokenizer.vocab), embedding_dim= 12, num_heads=4, context_length=32, num_layers = 8)
loaded_master_model.load_state_dict(torch.load("master_model.pth"))





new_tokens_3 = master_tokenizer.encode("each european country")


out = loaded_master_model.generate(tokens, 3)
print(out)

"""
probs3 = torch.softmax(out[-1], dim = -1)
max_probs3, max_index3 = torch.max(probs3, dim = -1)
print(max_probs3, max_index3, probs3) """




