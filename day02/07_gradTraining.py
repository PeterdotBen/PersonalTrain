import torch

# Defined a Tensor
x1 = torch.tensor([10, 20], requires_grad=True, dtype=torch.float32)
# Transform numpy array to x tensor
# It would have been better to use torch.from_numpy(x)
# happens RuntimeErrorL can't call numpy() on Tensor that requires grad. Use tensor.detach().numpy() instead. This means
# cannot make automatic differentiation of tensor Transforming numpy array to x tensor
# According to detach method to get new tensor as leaves node
x2 = x1.detach()
# x1 and x2 shared the same memory, but x2 can not automatically calculate gradient
print(x1.requires_grad)
print(x2.requires_grad)
# Those two are same value and share same memory
print(x1.data)
print(x2.data)
print(x1.is_leaf)
print(x2.is_leaf)
# make a tensor x2 transpose to numpy  array
print(x2.numpy())
