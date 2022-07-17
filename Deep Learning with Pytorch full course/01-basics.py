import numpy as np
import torch

x = torch.ones(2, 3, dtype=torch.float16)  # Generates simple tensor, pretty similar to Tensorflow
print(x)

x_built_with_list = torch.tensor([2.5, 10])
print(x_built_with_list)

#########################
# simple operations

x = torch.ones(2, 2)
y = torch.ones(2, 2)
z = torch.add(x, y)  # element wise sum
print(z)

########################
# Reshapes

x = torch.rand(4, 4)
y = x.view(16)
print(y)

#######################
# Converting from numpy to torch

a = np.ones(5)
print(a)
b = torch.from_numpy(a)
print(b)

a =+ 1
print(a)
print(b) # also b changed value, I guess this is due to the fact that it is not a copy of the original data, but rather a view


x = torch.ones(5, requires_grad=True) # enables gradient optimization for this variable/tensor