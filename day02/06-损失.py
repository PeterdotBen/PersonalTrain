"""
演示
    梯度下降法,求损失函数的最小值
    循环实现,计算梯度,更新参数w
题目:
    # 求loss = w**2 +20 的极小值,并打印loss是最小值时w的值(梯度)
计算过程:
    # 1 定义点w=10, require是_grad = True dtype = torch.float32
    # 2 定义函数loss = w**2 +20
    # 3 利用梯度小奖罚 循环迭代100 求最优解/最小值
        # 3.1 前向传播,计算损失
        # 3.2 梯度清零 w.grad.zero_()
        # 3.3 反向传播 loss.backward()
        # 3.4 更新参数 w.data = w.data -0.01 * w.grad
"""

import torch

# 1.定义点w=10, require是_grad = True dtype = torch.float32
w = torch.tensor([10.0, 20.0], requires_grad=True, dtype=torch.float32)
# 2 定义函数 loss = w**2+20
loss = w**2 + 20
# 3 利用梯度下降法, 循环迭代1000 求最优解
print(f"初始值:w:{w}, w.grad:{w.grad}, loss.mean():{loss.mean()}")
for i in range(1000):
    # 3.1 前向传播, 计算损失
    loss = w**2 + 20
    # 3.2 梯度清零 w.grad.zero_()
    if w.grad is not None:
        w.grad.zero_()
    # 3.3 反向传播 loss.backward()
    loss.mean().backward()
    # loss_mean' = 2*w
    print(f"第{i}次迭代:w:{w}, w.grad:{w.grad}, loss.mean():{loss.mean()}")
    # 3.4 梯度更新 w.data = w.data -0.01 * w.grad
    w.data = w.data - 0.01 * w.grad.data

