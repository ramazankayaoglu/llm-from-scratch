import torch

device = "cpu"

if torch.cuda.is_available():
    device = "cuda"
elif torch.backends.mps.is_available():
    device = "mps"
    
print(f"Using device: {device}")


from v2.master_model import MasterModel
from v2.master_tokenizer import MasterTokenizer
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor


#master_tokenizer = MasterTokenizer("mytokenizer.json")
master_tokenizer = MasterTokenizer("v2/tokenizer.json")

prompt = "important capitals, "

tokens = master_tokenizer.encode(prompt)
tokens = tokens.to(device)
print(tokens)


torch.manual_seed(1)
context_length = 32
master_model = MasterModel(
    vocab_size=len(master_tokenizer.vocab),
    embedding_dim= 12,
    num_heads=4,
    context_length=32,
    num_layers = 8,
    device = device)

out = master_model(tokens)
#print(out.shape)


with open("v2/text.txt", "r") as f:
    text = f.read()

#print(len(text), text[:100])

token_ids = master_tokenizer.encode(text)
#print(len(token_ids))

ids = token_ids.detach().cpu().numpy().tolist()
#print(len(ids), type(ids))


#from module1_1 import train_data_loader
from v1.text_dataset import TextDataset

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
"""
import torch.nn as nn

loss_fn = nn.CrossEntropyLoss()
loss = loss_fn(out0, dataset.targets[0])
#print(loss)

# optimizer = torch.optim.SDG(model.parameters(), lr = 1e-3) 
optimizer = torch.optim.AdamW(master_model.parameters(), lr = 1e-3)


epoch = 3000
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

loaded_master_model = MasterModel(vocab_size=len(master_tokenizer.vocab), embedding_dim= 12, num_heads=4, context_length=32, num_layers = 8, device = device)
loaded_master_model.load_state_dict(torch.load("v1/master_model.pth"))

ramos = loaded_master_model.generate(tokens, 1)
print(master_tokenizer.decode(ramos))
