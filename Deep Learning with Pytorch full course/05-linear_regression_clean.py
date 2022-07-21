import numpy
import torch
import torch.nn as nn
import numpy as np
from sklearn import datasets
import matplotlib.pyplot as plt

X_numpy, y_numpy = datasets.make_regression(n_samples=100, n_features=1, noise=20, random_state=1)

X = torch.from_numpy(X_numpy.astype(numpy.float32))
y = torch.from_numpy(y_numpy.astype(numpy.float32))  # I need to expand its dimension

y = y[..., None]  # https://stackoverflow.com/questions/68455417/torch-tensor-add-dimension
print(y.shape)
print(X.shape)


class LinearRegression(nn.Module):

    def __init__(self, input_dim, output_dim):
        super(LinearRegression, self).__init__()
        self.linear_layer = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        return self.linear_layer(x)


n_samples, n_features = X.shape

model = LinearRegression(n_features, 1)
loss_function = nn.MSELoss()
n_epochs = 400
learning_rate = 0.01

optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

for epoch in range(n_epochs):

    y_pred = model(X)

    loss = loss_function(y_pred, y)

    loss.backward()

    optimizer.step()

    optimizer.zero_grad()

    if (epoch + 1) % 10 == 0:
        print(f'epoch: {epoch + 1}, loss = {loss.item():.4f}')

# plot

predicted = model(
    X).detach()  # now Im using this tensor as result. I want to detach it from the graph. I'll generate a new tensor with req_grad = False

plt.plot(X_numpy, y_numpy, 'ro')
plt.plot(X_numpy, predicted, 'b')
plt.show()
