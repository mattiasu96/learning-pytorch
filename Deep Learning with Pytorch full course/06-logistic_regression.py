import torch
import torch.nn as nn
import numpy as np
from sklearn import datasets
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

bc_dataset = datasets.load_breast_cancer()
X, y = bc_dataset.data, bc_dataset.target

n_samples, n_features = X.shape
print(n_samples, n_features)

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1234)

standard_scaler = StandardScaler()

# The fit of the scaler computes the parameters based on the input data (so the mean and std deviation to normalize the data)
# then I transform the data once I computed such parameters. It is important to notice that I don't fit again the
# scaler on the test data! If I do it, I would be leaking info from the test data!
X_train = standard_scaler.fit_transform(X_train)
X_test = standard_scaler.transform(X_test)

X_train = torch.from_numpy(X_train.astype(np.float32))
X_test = torch.from_numpy(X_test.astype(np.float32))

y_train = torch.from_numpy(y_train.astype(np.float32))
y_test = torch.from_numpy(y_test.astype(np.float32))

y_train = y_train[..., None]
y_test = y_test[..., None]


class LogisticRegression(nn.Module):

    def __init__(self, input_dim, output_dim):
        super(LogisticRegression, self).__init__()
        self.linear = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        return torch.sigmoid(self.linear(x))


model = LogisticRegression(n_features, 1)

loss_function = nn.BCELoss()

optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

n_epochs = 100

for epoch in range(n_epochs):

    y_pred = model(X_train)

    loss = loss_function(y_pred, y_train)

    loss.backward()

    optimizer.step()

    optimizer.zero_grad()

    if (epoch + 1) % 10 == 0:
        print(f'epoch: {epoch + 1}, loss = {loss.item():.4f}')


# plot

with torch.no_grad(): # Again, it is needed to avoid tracking the gradient when I operate on such tensors
    y_predicted = model(X_test)
    y_pred_cls = y_predicted.round()
    acc = y_pred_cls.eq(y_test).sum() / float(X_test.shape[0]) # accuracy implemented by hand
    print(f'accuracy = {acc}')
