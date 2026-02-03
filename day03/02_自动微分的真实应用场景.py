import torch

torch.manual_seed(28)

# 1.定义输入数据x (2,5)

x = torch.ones(2, 5)
y = torch.zeros(2, 1)

# 3.初始化模型参数w和b
# z = x@w +b

w = torch.randn(5, 1, requires_grad=True)
b = torch.randn(1, requires_grad=True)

print(f'w:{w},shape:{w.shape},b:{b},b:{b.shape}')

z = x @ w + b

# 5.计算损失
# 5.1 先定义一个损失函数对象
loss_fn = torch.nn.MSELoss()
# 传入预测值和真实值
loss = loss_fn(z, y)
print(f'loss:{loss},shape:{loss.shape}')
# 6.梯度清零
# 7.自动求导
loss.backward()
print(f'w.grad:{w.grad},shape:{w.grad.shape}')
# 8.更新梯度
w.data = w.data - 0.01 * w.grad
print(f"w.grad:{w.grad},w.grad.shape:{w.grad.shape}")
print(f"b.grad:{b.grad},b.grad.shape:{b.grad.shape}")
