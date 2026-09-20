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

out = master_model(batch_tokens)
print(out.shape)




tokens.unsqueeze(0)
print(master_model.generate(tokens, 2))
tokens.shape

# save model
torch.save(master_model.state_dict(), "master_model.pth")

# load model
master_model.load_state_dict(torch.load("master_model.pth"))

