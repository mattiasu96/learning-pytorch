import torch

# This enables the gradient that will be computed in my output function y.
# y points to a grad_fn function that is used to compute the gradient with respect to x
x = torch.randn(3, requires_grad=True)
print(x)

y = x + 2
print(y)  # Here you can see tha grad_fn function
z = y * y * 2
print(z)  # I have again a grad_fn function associated with z. In this case it is a MulBackWard (backward for
# multiplication)

z = z.mean()  # In this case z becomes a scalar after the mean, thus I don't need a vector in the backward step

z.backward()  # dz/dx
print(x.grad, "printing grad")

####################

x = torch.randn(3, requires_grad=True)
print(x)

# This code gives an error if I don't specific anything in the backward step. Because I need a vector in there for the
# Jacobian.
y = x + 2
z = y * y * 2

v = torch.tensor([0.1, 1.0, 0.001], dtype=torch.float32)
z.backward(v)  # dz/dx
print(x.grad, "printing grad")


########################
# Prevent pytorch to save the history and calculate this grad_fn attribute. Prevent tracking the gradient

x = torch.randn(3, requires_grad=True)

# x.requires_grad_(False)
# x.detach()
# with torch.no_grad(): ....

# I can use the three above approaches to remove the gradient tracking.

#####################
# Dummy training example

weights = torch.ones(4, requires_grad=True)

for epoch in range(3):
    model_output = (weights*3).sum()
    model_output.backward()
    print("printing weight: ", weights.grad)
    weights.grad.zero_() # without this the gradient gets accumulated -> wrong result. I need to reset it at each iteration of the training

###############
# The detach and those other commands removes the variable from the computation of the gradient. Basically it becomes
# a constant. If an input variable has requires_grad = True, it means it is a differentiable variable and thus in the backward
# step I need to compute the derivative with respect to such variable.
# More here: https://www.kaggle.com/code/residentmario/pytorch-autograd-explained/notebook
# and here: https://www.youtube.com/watch?v=MswxJw-8PvE
# The accumulator is a method to optimize the computation of the gradient descent in mini-batches.
# More here: https://towardsdatascience.com/how-to-break-gpu-memory-boundaries-even-with-large-batch-sizes-7a9c27a400ce
# and here: https://towardsdatascience.com/what-is-gradient-accumulation-in-deep-learning-ec034122cfa

# il detach posso anche usarlo per detachare l'output, rimuovergli il grad_func ecc..., così che python mi fa il garbage collector
# del grafo di computazione e rimuove la roba non più utile che avevo usato nel training.