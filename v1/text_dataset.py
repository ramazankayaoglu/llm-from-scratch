import torch
from torch.utils.data import Dataset, DataLoader

pad_id = 63

class TextDataset(Dataset):
    def __init__(self, token_ids: list, context_length: int, stride: int):
        super().__init__() #Dataset alınır

        self.inputs = []
        self.targets = []

        for i in range(0, len(token_ids) - context_length, stride):
            input_chunk = token_ids[i: i + context_length]
            target_chunk = token_ids[i + 1: i + context_length + 1] #next token predict bu oluyor girdinin en baştaki token'i yerine bir sonraki token alınıp çıktı oluşturuluyor 

            #truncate the context length(tıraşlama denir)    
            input_chunk = input_chunk[:context_length] #context length uzunluğunu aşması durumunda context length kadar eleman alınır tıraşlama denir
            target_chunk = target_chunk[:context_length]

            #pad the context length
            input_chunk = input_chunk + [pad_id] * (context_length - len(input_chunk)) # burada da eğer context length kadar token yoksa boşluklar pad_id ile dolduruluyor
            target_chunk = target_chunk + [pad_id] * (context_length - len(target_chunk))

            #truncate the context length again(tekrar tıraşlama)    
            input_chunk = input_chunk[:context_length] 
            target_chunk = target_chunk[:context_length]

            self.inputs.append(torch.tensor(input_chunk, dtype = torch.long)) #tensor çok boyutlu liste 
            self.targets.append(torch.tensor(target_chunk, dtype = torch.long))



        
    def __len__(self):              #uzunluk almak için
        return len(self.inputs)

    def __getitem__(self, idx):     #istenilen elemanı almak için input ve target verir
        return self.inputs[idx], self.targets[idx]

stride = 1
context_length = 12


def create_data_loader(token_ids: list, context_length: int, stride: int, batch_size: int, shuffle: bool = False, device: str = "cpu"): #batch_size fine tuning yani düzeltme yapma sayısı  
    dataset = TextDataset(token_ids, context_length, stride)                                                                           #1 dersek her adımda düzeltme yapar 5 dersek 5 adımda 1 kez düzeltme yapar 
    dataloader = DataLoader(dataset, batch_size = batch_size, shuffle = shuffle, generator = torch.Generator(device = device))                 #device nerede işlem yapsın
                                                                                                                                       #Shuffle datayı her aldığnda tekrar tekrar karıştırsın mı 
                                                                                                                                       #generator dataloader'dan next item next item diye sıradaki veriyi alır
    return dataloader