import numpy as np

# f = w * x -> linear regression

# f = 2 * x
X = np.array([1, 2, 3, 4], dtype=np.float32)
Y = np.array([2, 4, 6, 8], dtype=np.float32)

w = 0.0


# model prediction
def forward(x):
    return w * x


# loss (Im using MSE)
def loss(y, y_predicted):
    return ((y_predicted - y) ** 2).mean()


# gradient
# MSE = 1/N * (w*x - y)^2
# dL/dw = 1/N 2x (w*x -y)
def gradient(x, y, y_predicted):  # NB: y_predicted = w*x
    return np.dot(2 * x, y_predicted - y).mean()  #


print(f'Prediction before training: f(5) = {forward(5):.3f}')

#training
learning_rate = 0.01
n_iters = 10

for epoch in range(n_iters):

    y_pred = forward(X)

    l = loss(Y, y_pred)

    dw = gradient(X, Y, y_pred)

    w -= learning_rate  * dw #updating the weight

    if epoch % 1 == 0:
        print(f'epoch {epoch+1}: w = {w: .3f}, loss = {l: .8f}')

print(f'Prediction after training: f(5) = {forward(5):.3f}')

# Results show how the loss decreases and the weight I am using moves from 1.2 to 2 as value, which is the expected one
# in the original function (I defined a y = 2*x)

