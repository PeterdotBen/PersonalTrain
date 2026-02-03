import torch
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

# 设置matplotlib中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']  # 设置中文字体
plt.rcParams['axes.unicode_minus'] = False  # 设置负号显示
# 1. 创建画布和坐标轴, 1行2列.
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
# 2. 生成 -20 ~ 20之间的 1000个数据点.
x = torch.linspace(-20, 20, 1000)
# 3. 上述1000个点, 输入sigmoid(x)
y = torch.sigmoid(x)
# 4. 在第1个子图中绘制sigmoid激活函数的图像.
axes[0].plot(x, y)
axes[0].set_title('sigmoid激活函数图像')
axes[0].grid()

# 5. 在第2个图上, 绘制sigmoid激活函数的导数图像.
# 重新生成 -20 ~ 20之间的 1000个数据点.
# 参1: 起始值, 参2: 结束值, 参3: 元素的个数, 参4: 是否需要求导.
x = torch.linspace(-20, 20, 1000, requires_grad=True)

# 上述1000个点, 输入sigmoid进行求导，sum().backward()
torch.sigmoid(x).sum().backward()

# 绘制图像
axes[1].plot(x.detach(), x.grad)
axes[1].set_title('sigmoid激活函数导数图像')
axes[1].grid()
plt.show()
