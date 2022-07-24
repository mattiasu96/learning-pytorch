import torch
import torch.nn as nn
import numpy as np

loss = nn.CrossEntropyLoss()  # NB: this layer applies all together softmax + negative log likelihood loss. So I dont
# need to apply anoyher softmax in my output layer!

# Y label must have the original labels, no one hot applied
Y = torch.tensor([2, 0, 1])  # nsamples x nclasses = 3x3
Y_pred_good = torch.tensor([[0.1, 1.0, 2.1], [2.0, 1.0, 0.1], [0.1, 3.0, 0.1]])  # No softmax applied yet, raw values
Y_pred_bad = torch.tensor([[2.1, 1.0, 0.1], [0.1, 1.0, 2.3], [0.5, 2.0, 0.3]])  # No softmax applied yet, raw values

l1 = loss(Y_pred_good, Y)
l2 = loss(Y_pred_bad, Y)

print(l1.item())
print(l2.item())

_, predictions1 = torch.max(Y_pred_good, 1)
_, predictions2 = torch.max(Y_pred_bad, 1)

print(predictions1)
print(predictions2)
