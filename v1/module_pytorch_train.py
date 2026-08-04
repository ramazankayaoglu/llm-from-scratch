import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor


training_data = datasets.FashionMNIST(
    root = "data",
    train = True,
    download = True,
    transform = ToTensor()
)

#Download test data from open datasets.
test_data = datasets.FashionMNIST(
    root = "data",
    train = False,
    download = True,
    transform = ToTensor()
)


#create a dataloader
from torch.utils.data import DataLoader

dataloader = DataLoader(training_data, batch_size = 64, shuffle = True)

#print(dataloader)


batch_size = 64 #

train_dataloader = DataLoader(training_data, batch_size = batch_size)
test_dataloader = DataLoader(test_data, batch_size)

#for X,y in test_dataloader:
    #print(f"Shape of X [N, C, H, W]: {X.shape}")
    #print(f"Shape of y: [N, C, H, W]: {y.shape} {y.dtype}")
    #break



import matplotlib.pyplot as plt

def show_data(data):    
    plt.imshow(data[0].squeeze(), cmap = "gray")
    plt.show()

#show_data(training_data[2][0])
#print(training_data[0][1])


device = torch.accelerator.current_accelerator.type if torch.accelerator.is_available() else "cpu" 
print(f"Using {device} device")


# Define model
class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(28*28, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 10)
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits


"""model = NeuralNetwork().to(device)
print(model)

print(torch.softmax(model(training_data[0][0]), dim = -1))"""

#show_data(training_data[0])

#count the number of parameters in the model
model = NeuralNetwork().to(device)
#print(sum(p.numel() for p in model.parameters()))


def train(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    model.train()
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)

        # Compute prediction error
        pred = model(X)
        loss = loss_fn(pred, y)

        # Backpropagation
        loss.backward()  #her bir parametrenin ne kadar güncelleneceğini hesaplar
        optimizer.step()  #parametreleri günceller
        optimizer.zero_grad()  #her bir parametrenin gradientını sıfırlar

        if batch % 100 == 0:
            loss, current = loss.item(), (batch + 1) * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")


def test(dataloader, model, loss_fn):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval() #evaluation mode'a alınır parametrelerin güncellenmesini durdurur
    test_loss, correct = 0, 0
    with torch.no_grad(): #gradient hesaplamasını durdurur
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            test_loss += loss_fn(pred, y).item() #loss fonksiyonunu hesaplar
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
    test_loss /= num_batches #test loss'u hesaplar
    correct /= size #correct'u hesaplar yani doğru tahminlerin sayısını hesaplar yani accuracy'yi hesaplar
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n") #test error'u ve accuracy'yi yazdırır


loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)


epochs = 5
for t in range(epochs):
    print(f"Epoch {t+1}\n-------------------------------")
    train(train_dataloader, model, loss_fn, optimizer)
    test(test_dataloader, model, loss_fn)
print("Done!")


print(torch.softmax(model(training_data[0][0]), dim = -1))