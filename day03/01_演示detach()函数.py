import torch

w = torch.tensor([1.0, 2.0, 3.0], requires_grad=True, dtype=torch.float32)
print(f'w:{w},shape:{w.shape},requires_grad:{w.requires_grad}')

t1 = w.detach()
print(f't1:{t1},shape:{t1.shape},type:{type(t1)},requires_grad:{t1.requires_grad}')
n2 = w.detach().numpy()
print(f"n2:{n2},shape:{n2.shape}")

n2[0]= 28.0
print(f"w:{w},shape{w.shape}")
print(f"n2:{n2},shape:{n2.shape}")