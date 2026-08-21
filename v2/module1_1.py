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


from tkinter import Y
import token

from plotly.graph_objs.layout import yaxis
from plotly.graph_objs.layout.scene import zaxis
from sympy.geometry.plane import y
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


from text_dataset import create_data_loader

stride = 1
context_length = 12



train_data_loader = create_data_loader(token_ids, context_length, stride, 1)
#print(len(train_data_loader))

i = 0
for batch in enumerate(train_data_loader):
    #print(batch)
    i += 1
    if i > 3:
        break

data_iter = iter(train_data_loader)
#print(next(data_iter))


##Embedding is the numerical representation of the dictionary meanings(Embedding yani Sözlük Anlamlarının Sayısal Değerleri)

dict_v2 = {
    "the" : [0.1, 0.0, 0.0, 0.0],
    "capital": [0.4, 0.2, 0.04, 0.3],
    "of": [0.0, 0.24, 0.02, 0.01],
    "united": [0.0, 0.0, 0.03, 0.01]
}

#print(dict_v2["the"], dict_v2["capital"], dict_v2["of"], dict_v2["united"])

import plotly.graph_objects as go
import plotly.offline

def plot_dots(sentences_data, title, dims=[0, 1, 2]):
    data = [
        go.Scatter3d(
            x = sentences_data["words"][:, dims[0]],
            y = sentences_data["words"][:, dims[1]],
            z = sentences_data["words"][:, dims[2]],
            mode = "markers+text",
            marker=dict(
                size = 6,
                color = sentences_data["color"]
            ),
            text = sentences_data["labels"],
            hoverinfo="text"
        )for sentences_data in sentences_data
    ]
    layout = go.Layout(
    scene = dict(
        xaxis_title="Sertlik",
        yaxis_title="Parlaklık",
        zaxis_title="Kırmızılık"
    ),
    title = "title")

    fig = go.Figure(data = data, layout = layout)
    plotly.offline.iplot(fig)

import numpy as np


sentences = [
    {
        "words": np.array([
            [0.1, 0.20, 0.23, 0.0],
            [0.4, 0.2, 0.04, 0.3],
            [0.12, 0.24, 0.02, 0.01],
            [0.0, 0.0, 0.3, 0.01],
            [0.0, 0.0, 0.0, 0.0],
        ]),
        "labels": ["the", "capital", "of", "united", "states"],
        "color": "red"
    }
]


#plot_dots(sentences, "Sözlük V1")

import torch

embeddings = torch.nn.Embedding(num_embeddings=64, embedding_dim=4)


from master_tokenizer import MasterTokenizer
tokenizer = MasterTokenizer("tokenizer.json")

ids = tokenizer.encode(text)

tokens = tokenizer.encode("the capital of united states and the capital of france")
#print(tokens)

meanings =embeddings(torch.tensor(tokens))
#print(meanings.shape)

sentences = [
    {
        "words": meanings.detach().numpy(),
        "labels": ["the", " ", "capital", " ", "of", " ", "united", " ", "states", " ", "the", " ", "capital", " ", "of", " ", "france"],
        "color": "red"
    }
]

plot_dots(sentences, "Sözlük V1")



embeddings = torch.nn.Embedding(num_embeddings=64, embedding_dim=4)
sentence = "the capital of united states and the capital of france"
tokens = tokenizer.encode(sentence)
meanings = embeddings(torch.tensor(tokens))

#print(gemma_tokenizer.tokenize(sentence))

gemma_sentences = [
    {
        "words": meanings.detach().numpy(),
        "labels": tokenizer.tokenize(sentence),
        "color": "red"
    }
]
#plot_dots(gemma_sentences, "Gemma V1")

from transformers import AutoTokenizer, AutoModelForCausalLM, Gemma3ForCausalLM

gemma_tokenizer = AutoTokenizer.from_pretrained("google/gemma-3-1b-it")
gemma_model = AutoModelForCausalLM.from_pretrained("google/gemma-3-1b-it")
#model2 = Gemma3ForCausalLM.from_pretrained("google/gemma-3-1b-it")

gemma_tokens = gemma_tokenizer.encode("the capital of united states the capital of france") #burada görselleştirdiğimizde capital, the ve of kelimeleri bizim
#print(gemma_tokens)                                                                         #sözlüğümüzden farklı olarak 2 tane gözükür ve tam olarak aynı nokta
#print(gemma_model)                                                                         #üzerinde değildir çünkü gemma 

#print(gemma_model.model.embed_tokens(torch.tensor(gemma_tokens)))

gemma_meanings = gemma_model.model.embed_tokens(torch.tensor(gemma_tokens))
#print(gemma_meanings.shape)

#print(gemma_tokenizer.tokenize("the capital of united states"))

print(gemma_tokenizer.tokenize(sentence))

gemma_sentences = [
    {
        "words": gemma_meanings.detach().float().numpy(),
        "labels": gemma_tokenizer.tokenize(sentence),
        "color": "red"
    }
]
#plot_dots(gemma_sentences, "Gemma V1", dims=[20, 21, 22])


"""
data = [
    go.Scatter3d(
        x = [0.1, 0.4, 0.13, 0.0, 0.0],
        y = [1.0, 0.2, 0.94, 0.0, 0.0],
        z = [0.9, 0.2, 0.87, 0.03, 0.0],
        mode="markers+text",
        marker=dict(
            size = 10,
            color = "red"
        ),
        text = ["the", "capital", "of", "united", "states"], 
        hoverinfo="text"
    )
]

layout = go.Layout(
    scene = dict(
        xaxis_title="Sertlik",
        yaxis_title="Parlaklık",
        zaxis_title="Kırmızılık"
    ),
    title = "Sözlük V1"
)

fig = go.Figure(data=data, layout = layout)
plotly.offline.iplot(fig)
"""