import torch

from master_model import MasterModel
from master_tokenizer import MasterTokenizer

device = "cpu"

if torch.cuda.is_available():
  device = "cuda"
elif torch.backends.mps.is_available():
  device = "mps"
  

print(f"Using device: {device}")

master_tokenizer = MasterTokenizer("v2/tokenizer.json")
"""
prompts = [
  "the capital of the united",
  "madrid is in",
  "the capital of france is",
  "the capital of germany is"
]

tokens = master_tokenizer.encode(prompts[0])
tokens = tokens.to(device)
print(tokens)
batch_tokens = master_tokenizer.encode_batch(prompts, 32)
batch_tokens = batch_tokens.to(device)
batch_tokens.shape
"""

torch.manual_seed(1)
context_length = 32
torch.manual_seed(1)
master_model = MasterModel(
  vocab_size=len(master_tokenizer.vocab),
  embedding_dim=12,
  num_heads=4,
  context_length=context_length,
  num_layers=8,
  device=device
)
"""
out = master_model(batch_tokens)
out.shape


out.flatten(0, 1).shape


out = master_model.generate(tokens, 3)
master_tokenizer.decode(out)


with open("v2/text.txt", "r") as f:
  text = f.read()

len(text), text[:100]


token_ids = master_tokenizer.encode(text)
len(token_ids), type(token_ids)


from text_dataset import create_data_loader

stride = 12


train_data_loader = create_data_loader(token_ids.tolist(), context_length, stride, 16, False)

len(train_data_loader)


# model parameters count
parameters_count = sum(p.numel() for p in master_model.parameters())
print(parameters_count)

# model architecture
print(master_model)
"""
"""
import torch.nn as nn

loss_fn = nn.CrossEntropyLoss()


# optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)
optimizer = torch.optim.AdamW(master_model.parameters(), lr=1e-3)


for i, (X, Y) in enumerate(train_data_loader):
  print(X.shape, Y.shape, Y.flatten().shape)
  break
"""
"""
epoch = 30000

for epoch in range(epoch):
  total_loss = 0.
  for i, (X, Y) in enumerate(train_data_loader):
    X = X.to(device)
    Y = Y.to(device)
    
    pred = master_model(X)
    loss = loss_fn(pred.flatten(0, 1), Y.flatten())
    total_loss += loss.item()
    
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
    
  average_loss = total_loss / len(train_data_loader)
  print(f"Epoch {epoch + 1} loss: {loss.item()} average loss: {average_loss}")


"""
"""
import torch

new_tokens = master_tokenizer.encode("the capital of the united states is")
new_tokens = new_tokens.tolist()
# new_tokens.append(61)

out = master_model(torch.tensor([new_tokens]).to(device))
out = out.squeeze(0)
probs = torch.softmax(out[-1], dim=-1)
max_prob, max_index = torch.max(probs, dim=-1)
max_prob, max_index, probs
"""


# save model
torch.save(master_model.state_dict(), "v2/master_model.pth")

# load model
master_model.load_state_dict(torch.load("v2/master_model.pth"))

"""
# generate text
new_tokens = master_tokenizer.encode("the capital of the united states is london. the capital of france is")
new_tokens = new_tokens.detach().cpu().numpy().tolist()
new_tokens.append(61)
len(new_tokens)
"""


loaded_model = MasterModel(64, embedding_dim=12, num_heads=4, context_length=32, num_layers=8, device=device)
loaded_model.load_state_dict(torch.load("v2/master_model.pth"))
loaded_model



"""
out = master_model(torch.tensor(new_tokens).unsqueeze(0).to(device))

probs = torch.softmax(out[-1], dim=-1)
max_prob, max_index = torch.max(probs, dim=-1)
max_prob, max_index, probs
"""


import torch

new_tokens = master_tokenizer.encode("the capital of the united kingdom is")
new_tokens = new_tokens.detach().cpu().numpy().tolist()
new_tokens.append(61)

print(master_tokenizer.decode(master_model.generate(torch.tensor(new_tokens), 9)))