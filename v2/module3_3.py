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


torch.manual_seed(1)
context_length = 32


master_model = MasterModel(
  vocab_size=len(master_tokenizer.vocab),
  embedding_dim=12,
  num_heads=4,
  context_length=context_length,
  num_layers=8,
  device=device
)

# load model
master_model.load_state_dict(torch.load("v2/master_model.pth"))

out = master_model(batch_tokens)
out.shape



#temperature : sıcaklık
#top_k: k adet en yüksek olasılıklı token seçer
#top_p: next prediction'da olasılıkların toplamı normalde 1 eder, burada olasılıklarının toplamı 0.95 0.90 gibi ayarlamaların yapıldığı terim


"""outs = {}
for _ in range(1000):
  out = master_model.generate(tokens, 3, temperature = 0.50)
  decoded = master_tokenizer.decode(out)
  outs[decoded] = outs.get(decoded, 0) + 1
print(outs) """



top_k = 10

sorted_outs = sorted(out[-1][-1].tolist(), reverse=True)
sorted_indexes = []
for so in sorted_outs[:top_k]:
  so_index = out[-1][-1].tolist().index(so)
  sorted_indexes.append(so_index)
sorted_outs = torch.tensor(sorted_outs[:top_k])
print(sorted_outs, sorted_indexes)