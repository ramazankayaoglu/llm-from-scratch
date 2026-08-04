import torch 
import torch.nn as nn

class MasterLayerNormalization(nn.Module):
    def __init__(self, embedding_dim, eps = 1e-5):
        super().__init__()
        self.eps = eps
        
        self.weight = nn.Parameter(torch.ones(embedding_dim)) #Normalizasyonun ağırlıkları default olarak 1 ancak süreç içinde Network
                                                              #içinde değişebilir Parameter ise verilen Dim'de Matris oluşturur

    def forward(self, x):
        mean = x.mean(dim = -1, keepdim = True)
        variance = x.var(dim = -1, keepdim = True, unbiased = False)
        normalized_x = (x - mean) / torch.sqrt(variance + self.eps)
        
        return self.weight * normalized_x   