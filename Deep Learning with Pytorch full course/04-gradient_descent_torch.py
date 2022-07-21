import torch
import torch.nn as nn

# 1 ) Design model (input, output, forward step)
# 2 ) Contruct loss and optimizer
# 3 ) Training loop:
#     - forward pass: compute prediction
#     - backward pass: gradients
#     - update weights

# f = w * x -> linear regression

# f = 2 * x
X = torch.tensor([[1], [2], [3], [4]],
                 dtype=torch.float32)  # NB the order of batch, sampple, features is different from tensorflow
Y = torch.tensor([[2], [4], [6], [8]], dtype=torch.float32)

X_test = torch.tensor([5], dtype=torch.float32)
n_sample, n_features = X.shape
print(n_sample, n_features)

# w = torch.tensor(0.0, dtype=torch.float32, requires_grad=True)

input_size = n_features
output_size = n_features
model = nn.Linear(input_size, output_size)


# Example Custom model by inheriting from the nn.Module
class LinearRegression(nn.Module):

    def __init__(self, input_dim, output_dim):
        super(LinearRegression, self).__init__()
        # define layers
        self.linear_layer = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        return self.linear_layer(x)


model = LinearRegression(input_size, output_size)

# model prediction
# def forward(x):
#     return w * x


loss = nn.MSELoss()

print(f'Prediction before training: f(5) = {model(X_test).item():.3f}')

# training
learning_rate = 0.01
n_iters = 200
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

for epoch in range(n_iters):

    # y_pred = forward(X)

    y_pred = model(X)

    l = loss(Y, y_pred)

    l.backward()  # dl/dw, the gradient is on the loss with respect to the variables. In my case is W and has gradient to True

    # This part is important. The no_grad is needed to detach the tensor from the computational graph, aka setting
    # a copy with requires_grad = False. But why? Well, due to the functioning of the autograd, each operation on a variable
    # with requires grad = True is tracked and used for backprop. Thus, if you dont detach the tensor, also this operation
    # is tracked (updating the weight) and it will be used to compute the next backpropagation step. Which is obviously
    # wrong: the update of the weights does not belong to the operations performed by the network in the forward step!
    # with torch.no_grad():
    #    w -= learning_rate * w.grad  # updating the weight. This part should not be part of the computational graph

    optimizer.step()  # This automatically updates weight w, I pass the list of weights at initialization time

    # w.grad.zero_()  # reset the gradient of the variable

    optimizer.zero_grad()

    if epoch % 1 == 0:
        [w, b] = model.parameters()
        print(f'epoch {epoch + 1}: w = {w[0][0].item(): .3f}, loss = {l: .8f}')

print(f'Prediction after training: f(5) = {model(X_test).item():.3f}')

# Results show how the loss decreases and the weight I am using moves from 1.2 to 2 as value, which is the expected one
# in the original function (I defined a y = 2*x)
