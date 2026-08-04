import tiktoken
import json


#text1 = "The cat chased the dog"
#text2 = "The dog chased the cat"
#text = "The capital of France is"

#basic tokenization
#def tokenize(text):
 #   return text.split()

#def tokenize2(text):
    #parts = text.split()
    #ids = []
    #for part in parts:
        #if part in vocab:
        #    value = vocab[part]
       # else:
      #      value = vocab["<unk>"]
     #   ids.append(value)
    #return ids


#def detokenize(ids):
 #   text = ""
  #  for id in ids:
   #     part = reverse_vocab[id]
    #    text += part + " "
    #return text.strip()


#with open("tokenizer.json", "r") as f:
 #   vocab = json.load(f)


#reverse_vocab = {id: part for part, id in vocab.items()}



#print(tokenize(text1))
#print(tokenize2(text1))
#print(reverse_vocab)

#token_ids = tokenize2(text1)
#print(detokenize(token_ids))


#enc = tiktoken.get_encoding("gpt2")
#print(enc.encode(text1))
#gpt2_ids = enc.encode(text1)
#print(gpt2_ids)
#print(enc.decode(gpt2_ids))
#print(enc.n_vocab)


#enc2 = tiktoken.get_encoding("cl100k_base")
#print(enc2.n_vocab)
#enc3 = tiktoken.get_encoding("o200k_base")
#print(enc3.n_vocab)

#from huggingface_hub import login


# Load model directly
#from transformers import AutoProcessor

#processor = AutoProcessor.from_pretrained("google/gemma-3-27b-it")

#print(processor.tokenizer.encode(text1))
#print(processor.tokenizer.vocab_size)


#print(processor.tokenizer.encode(text1))
#print(processor.tokenizer.vocab_size)

#gemma_ids = processor.tokenizer.encode(text1)
#print(gemma_ids)
#print(processor.tokenizer.decode(gemma_ids))#beginning of the sequence da gelir başta <bos> olarak

#processor.tokenizer.save_pretrained("gemma_tokenizer")

#with open("tokenizer_gemma.json","w", encoding="utf-8") as f:
 #   json.dump(processor.tokenizer.get_vocab(), f, ensure_ascii= False) 

"""
from tokenizer import Tokenizer

tokenizer = Tokenizer("tokenizer.json")
print(tokenizer.encode("states"))
print(tokenizer.decode([4,58]))
"""


from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import Whitespace

tokenizer = Tokenizer(BPE(unk_token="<UNK>"))
tokenizer.pre_tokenizer = Whitespace()

trainer = BpeTrainer(
    vocab_size=30000,
    special_tokens=["<PAD>", "<UNK>", "<BOS>", "<EOS>"]
)

tokenizer.train(["text.txt"], trainer)
tokenizer.save("mytokenizer.json")

encoded = tokenizer.encode("merhaba kanka nasılsın")
print(encoded.tokens)
print(encoded.ids)