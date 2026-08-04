import sentencepiece as spm

#spm.SentencePieceTrainer.Train(
    #input = "text.txt",
   # model_prefix = "spm_tokenizer",
  #  vocab_size = 64,
 #   model_type = "bpe"
#)

text1 = "The cat chased the dog"
text2 = "The dog chased the cat"    

"""spm_tokenizer = spm.SentencePieceProcessor(model_file = "spm_tokenizer.model")

spm_ids = spm_tokenizer.Encode(text1)
spm_tokens = spm_tokenizer.Encode(text1, out_type=str)
print(spm_ids, spm_tokens) """


from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import Whitespace

hf_tokenizer = Tokenizer(BPE())
print(hf_tokenizer.get_vocab_size())
hf_tokenizer.pre_tokenizer = Whitespace()

trainer = BpeTrainer(vocab_size=64, special_tokens=["<unk>"])
hf_tokenizer.train(["text.txt"], trainer)

print(hf_tokenizer.get_vocab_size(), hf_tokenizer.encode(text1).ids)

hf_tokenizer.save("hf_tokenizer.json")

from transformers import PreTrainedTokenizerFast #Fast kütüphanesi çok daha hızlı train yapan rust diliyle yazılmış kütüphaneyi import eder python kodu
                                                 #ile çalıştırılan train kütüphanesinden çok daha hızlı şekilde train yapar büyük metinlerde kullanılmalı
fast_tokenizer = PreTrainedTokenizerFast(tokenizer_file = "hf_tokenizer.json")
print(fast_tokenizer.encode(text1))


print(fast_tokenizer.push_to_hub("ramosvaldo9/hf_tokenizer"))