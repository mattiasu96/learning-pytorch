import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
import time
start_time = time.time()
input_size = 784  # 28x28
hidden_size = 100
num_classes = 10
num_epochs = 2
batch_size = 100
learning_rate = 0.001

device = torch.device('mps')

train_dataset = torchvision.datasets.MNIST(root='./data', train=True, transform=transforms.ToTensor(), download=True)
test_dataset = torchvision.datasets.MNIST(root='./data', train=False, transform=transforms.ToTensor(), download=False)

train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=True)

examples = iter(train_loader)
samples, labels = examples.next()

for i in range(6):
    plt.subplot(2, 3, i + 1)
    plt.imshow(samples[i][0], cmap='gray')


class NeuralNet(nn.Module):

    def __init__(self, input_size, hidden_size, num_classes):
        super(NeuralNet, self).__init__()
        self.l1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.l2 = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        out = self.l1(x)
        out = self.relu(out)
        out = self.l2(out)  # Again, no softmax layer here, the crossentropy in pytorch already includes it
        return out


model = NeuralNet(input_size=input_size, hidden_size=hidden_size, num_classes=num_classes)
model = model.to(device)
loss_func = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
n_total_steps = len(train_loader)

for epoch in range(num_epochs):
    for i, (images, labels) in enumerate(train_loader):
        # image shape = 100, 1, 28, 28 -> 100 images of size 28x28x1
        # I am using a feed forward neural network, I need to flatten and reshape those images

        flatten_images = images.reshape(-1, 28 * 28).to(device)
        labels = labels.to(device)
        y_pred = model(flatten_images)
        loss = loss_func(y_pred, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (i+1) % 100 == 0:
            print(f'epoch {epoch+1} / {num_epochs}, step {i+1}, loss = {loss.item():.4f}')

# test

with torch.no_grad(): # I am testing, I don't want any gradient to be changed or updated!
    n_correct = 0
    n_samples = 0
    for images, labels in test_loader:
        flatten_images = images.reshape(-1, 28 * 28).to(device)
        labels = labels.to(device)
        outputs = model(flatten_images)

        # torch.max() returns (value, index)
        _, predictions = torch.max(outputs, 1)
        n_samples += labels.shape[0]
        n_correct = (predictions == labels).sum().item()

    acc = 100 * n_correct / n_samples
    print('accuracy:', acc)


print('elapsed_time:', time.time() - start_time)