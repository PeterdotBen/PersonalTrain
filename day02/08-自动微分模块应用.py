import torch

# 输入张量2*5
x = torch.ones(2, 5)
# 目标值是2*3
y = torch.zeros(2, 3)
# 设置要更新的权重和偏置初始值
w = torch.randn(5, 3, requires_grad=True)
b = torch.randn(3, requires_grad=True)
# 设置网络输出值
z = torch.matmul(x, w) + b  # 矩阵乘法
# 损失函数,并进行损失计算
loss = torch.nn.MSELoss()
loss = loss(z, y)
# 自动微分
loss.backward()
# 打印 w,b变量的梯度
# backward 函数计算的梯度值会存储在张量的grad变量中
print('W的梯度:', w.grad)
print('b的梯度', b.grad)
