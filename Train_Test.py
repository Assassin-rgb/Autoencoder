import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from AutoEncoder import AutoEncoder
import matplotlib.pyplot as plt
import numpy as np

batch_size = 64
epochs = 10
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# training function
def train(model, data, epoch, criterion, optimizer, model_number):
    print('\n--------------training start--------------')
    model.train()
    for i in range(epoch):
        epoch_loss = 0
        for x, _ in data:
            if model_number == 1:
                x = x.view(-1, 784)
            x = x.to(device)
            optimizer.zero_grad()
            out = model(x)
            loss = criterion(out, x)
            epoch_loss += loss.data
            loss.backward()
            optimizer.step()
        epoch_loss = epoch_loss/len(data)
        print(f'Epoch : {i+1}  Loss : {epoch_loss:.5f}')
    print('--------------training end--------------')


# testing function
def test(model, data, criterion, model_number):
    print('\n--------------evaluation start--------------')
    model.eval()
    result = []
    test_loss = 0
    for x, y in data:
        if model_number == 1:
            x = x.view(-1, 784)
        x = x.to(device)
        out = model(x)
        loss = criterion(out, x)
        test_loss += loss
        if model_number == 1:
            x = x.view(28, 28)
            out = out.view(28, 28)
        else:
            x = x.squeeze()
            out = out.squeeze()
        x = x.cpu().numpy()
        out = out.cpu().detach().numpy()
        result.append((x, out, y))
    test_loss = test_loss / len(data)
    print(f'Test Loss : {test_loss:.5f}')
    print('--------------evaluation end--------------')
    return result


# get parameters
def get_parameters(model):
    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    params = sum([np.prod(p.size()) for p in model_parameters])
    return params


# plot and save 20 samples from test data
def plot_save(data, model_number):
    total = 1
    classes = [0]*10
    for x, out, y in data:
        if total <= 20 and classes[y] != 2:
            plt.subplot(1, 2, 1)
            plt.title('Original Image')
            plt.imshow(x, cmap='gray')
            plt.subplot(1, 2, 2)
            plt.title('Reconstructed Image')
            plt.imshow(out, cmap='gray')
            plt.savefig(f'model{model_number}/{total}.png')
            total += 1
            classes[y] += 1
    print('\n--------------plots saved--------------')


# dataset
transform = transforms.Compose([transforms.ToTensor()])
train_data = datasets.MNIST(root='data', train=True, transform=transform, download=True)
train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
test_data = datasets.MNIST(root='data', train=False, transform=transform, download=True)
test_loader = DataLoader(test_data, batch_size=1)

# models
model1 = AutoEncoder(mode=1).to(device)
model2 = AutoEncoder(mode=2).to(device)
mse = torch.nn.MSELoss()
adam1 = torch.optim.Adam(model1.parameters(), lr=1e-3)
adam2 = torch.optim.Adam(model2.parameters(), lr=1e-3)

# training, testing and plotting model1
print('\n-------------------Model-1-------------------')
print(model1)
params1 = get_parameters(model1)
print('\nTotal Parameters : ', params1)
train(model1, train_loader, epochs, mse, adam1, 1)
out1 = test(model1, test_loader, mse, 1)
plot_save(out1, 1)

# training, testing and plotting model2
print('\n-------------------Model-2-------------------')
print(model2)
params2 = get_parameters(model2)
print('\nTotal Parameters : ', params2)
train(model2, train_loader, epochs, mse, adam2, 2)
out2 = test(model2, test_loader, mse, 2)
plot_save(out2, 2)
