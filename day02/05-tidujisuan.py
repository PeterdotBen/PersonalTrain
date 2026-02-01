import torch

# 1.定义变量,模拟模型权重参数w
w = torch.tensor([10,20],requires_grad=True,dtype=torch.float32)
print(f"w:{w}, requires_grad: {w.requires_grad}")
# 2.定义函数，模拟损失函数
loss = w**2 + 10    # [110,410]
loss_mean = loss.mean()
print(f"loss_mean:{loss_mean}")
# loss' = 2*w = [2*w1,2*w2]
# loss_mean' = 2*w = [w1,w2]
# 3.需要执行反向传播来调用自动微分模块
loss.mean().backward()
# 4.打印loss_mean对w的梯度
print(f"w.grad:{w.grad}")
# 5.更新参数：w新 = w旧 - 学习率 * 梯度
w.data = w.data - 0.01 * w.grad.data
# 10 - 0.01*20=9.8
# [10,20] - 0.01*[10.,20.]=[9.9,19.8]
print(f"w:{w}")