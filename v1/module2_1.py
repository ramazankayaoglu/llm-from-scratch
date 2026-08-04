from .master_model import MasterModel
from .master_tokenizer import MasterTokenizer
import torch

master_tokenizer = MasterTokenizer("tokenizer.json")

prompt = "the capital of united states and the capital of france"

tokens = master_tokenizer.encode(prompt).long()

torch.manual_seed(1)
master_model = MasterModel(
    vocab_size=len(master_tokenizer.vocab),
    embedding_dim=4,
    context_length=32
)
"""
master_model(tokens)
print(master_model(tokens))
print(master_model)
"""

sentence_meanings = master_model(tokens)
#print(sentence_meanings.shape)

"""
from transformers import  AutoModelForCausalLM
gemma_model = AutoModelForCausalLM.from_pretrained("google/gemma-3-1b-it")
print(gemma_model)"""


from plot_tokens import plot_tokens

master_sentences = [
    {
    "words" : sentence_meanings.detach().numpy(),
    "labels": master_tokenizer.tokenize(prompt),
    "color": "red"
    }
]

#print(u_tokenizer.tokenize(prompt))

#plot_tokens(master_sentences, "Models Context Space")

#print(sentence_meanings)


the_position = [-1.5256, -0.7502,  0.6540,  1.6095]
capital_position = [ 0.9326, -0.2774, -0.4988,  1.4560]

hardness_distance = abs(the_position[0] - capital_position[0])
brightness_distance = abs(the_position[1] - capital_position[1])
redness_distance = abs(the_position[2] - capital_position[2])
blueness_distance = abs(the_position[3] - capital_position[3])

#print(hardness_distance, brightness_distance, redness_distance, blueness_distance)

total_distance = hardness_distance + brightness_distance + redness_distance + blueness_distance #manhattan distance olarak geçer


apple = [-1.5256, -0.7502, -0.6540, -1.6095]
real_apple = [0.5, -0.7502, -0.6540, -1.6095]

def is_apple(position, real_position):
    dist1 = position[0] - real_position[0]

    print(dist1)

    #burada eğer değer doğru değilse tuning yani eğitim vererek doğru değere ulaştırmaya çalışıyor
    if dist1 > 0:
        apple[0] -= 0.5
    else:
        apple[0] += 0.5

    return dist1 > 0 and dist1 < 0.5


"""for i in range(1,5):
    print(is_apple(apple, real_apple))
    if(is_apple(apple, real_apple) == True):
        print(is_apple(apple, real_apple))
    else:
        is_apple(apple, real_apple)"""


cos_sim_hardness = the_position[0] * capital_position[0]
cos_sim_brightness = the_position[1] * capital_position[1]
cos_sim_redness = the_position[2] * capital_position[2]
cos_sim_blueness = the_position[3] * capital_position[3]

total_cos_sim = cos_sim_blueness + cos_sim_brightness + cos_sim_hardness + cos_sim_redness

#print(total_cos_sim)
#print(sentence_meanings[0], sentence_meanings[1], sentence_meanings[2], sentence_meanings[3])

c_0_0 = sentence_meanings[0][0] *  sentence_meanings[0][0] + sentence_meanings[0][1] * sentence_meanings[0][1] + sentence_meanings[0][2] * sentence_meanings[0][2] + sentence_meanings[0][3] * sentence_meanings[0][3]
c_0_1 = sentence_meanings[0][0] *  sentence_meanings[1][0] + sentence_meanings[0][1] * sentence_meanings[1][1] + sentence_meanings[0][2] * sentence_meanings[1][2] + sentence_meanings[0][3] * sentence_meanings[1][3]
c_0_2 = sentence_meanings[0][0] *  sentence_meanings[2][0] + sentence_meanings[0][1] * sentence_meanings[2][1] + sentence_meanings[0][2] * sentence_meanings[2][2] + sentence_meanings[0][3] * sentence_meanings[2][3]
c_0_3 = sentence_meanings[0][0] *  sentence_meanings[3][0] + sentence_meanings[0][1] * sentence_meanings[3][1] + sentence_meanings[0][2] * sentence_meanings[3][2] + sentence_meanings[0][3] * sentence_meanings[3][3]

#print(c_0_0, c_0_1, c_0_2, c_0_3)
"""
the_similarities = []

for i in range(len(sentence_meanings)): 
    cs_the_i = sentence_meanings[0][0] *  sentence_meanings[i][0] + sentence_meanings[0][1] * sentence_meanings[i][1] + sentence_meanings[0][2] * sentence_meanings[i][2] + sentence_meanings[0][3] * sentence_meanings[i][3]
    the_similarities.append(cs_the_i)

print(the_similarities)"""    


#aşağıdaki kod ile cosinus benzerliğinden kelimelerin yani Token'ların diğer Token'lara benzerliğini ölçmeye "Attention ağırlığı" denir
all_similarities = torch.zeros(sentence_meanings.shape[0], sentence_meanings.shape[0])
for i in range(sentence_meanings.shape[0]):
    i_similarities = torch.zeros(sentence_meanings.shape[0])

    for j in range(sentence_meanings.shape[0]):
        for k in range(sentence_meanings.shape[1]):
            cs_i_j = sentence_meanings[i][k] * sentence_meanings[j][k]
            i_similarities[j] = cs_i_j

    all_similarities[i] = i_similarities

#print(all_similarities.detach().numpy())

#Query, Key, Value
all_similarities_torch = sentence_meanings @ sentence_meanings.T  #sentence_meanings [4,20] boyutlarında bir matris bu matrisin Transpoz'u alınıp 
#print(sentence_meanings)                                          #[20,4] boyutlarındaki hali çarpıldığında istenilen tüm boyutların birbiri ile çarpılması
#print(sentence_meanings.T)                                        #yapılmış olur. Torch bunu aynı zamanda CPU'da multithread yaparak performans sağlar


#print(sentence_meanings @ sentence_meanings.T)

attention_weights = torch.softmax(all_similarities, dim = 1)
#print(attention_weights)

#print(torch.sum(all_similarities_torch[0]))
#print(torch.sum(attention_weights[0]))


sentence_context_vector = attention_weights @ sentence_meanings #sentence_meanings burada value oluyor 
#print(sentence_context_vector)
#print(sentence_meanings)

sentence_meanings_without_pos = master_model.embedding(tokens)

master_sentences = [
    {
    "words" : sentence_meanings_without_pos.detach().numpy(), #sözlükteki yeri
    "labels": master_tokenizer.tokenize(prompt),
    "color": "blue"
    },
    {
    "words" : sentence_meanings.detach().numpy(), #sıra bilgisi de eklendi
    "labels": master_tokenizer.tokenize(prompt),
    "color": "purple"
    },    
    {
    "words" : sentence_context_vector.detach().numpy(), #token bu cümlenin içinde ne kadar önemli ve anlam da eklendi 
    "labels": master_tokenizer.tokenize(prompt),
    "color": "orange"
    }
]

plot_tokens(master_sentences, "Models Attention Sentence Space")
