import torch
# numpy对象->张量->数据集对象TensorDataset->数据加载器DataLoader
from torch.utils.data import TensorDataset  # 够着数据集对象
from torch.utils.data import DataLoader  # 数据加载器, 按批次加载数据
from torch import nn  # 提供MSE的损失函数和线性层,线性层用来模拟线性回归模型
from torch import optim  # 提供各种优化器,用于更新模型参数, 公式为: W<-W-lr* grad
from sklearn.datasets import make_regression  # 创建线性回归示例数据集
import matplotlib

matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

# 设置matplotlib中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']  # 设置中文字体
plt.rcParams['axes.unicode_minus'] = False  # 设置负号显示


# 1.定义函数,创建线性回归数据
def create_dataset():
    # 创建数据集对象
    x, y, coef = make_regression(
        n_samples=100,  # 样本数量
        n_features=1,  # 特征数量
        n_targets=1,  # 1个目标变量
        noise=10,  # 噪声
        bias=13.98,  # 偏置
        coef=True,  # 是否返回系数
        random_state=6  # 随机种子
    )
    # 封装成张量
    x = torch.tensor(x, dtype=torch.float32)
    y = torch.tensor(y, dtype=torch.float32)
    return x, y, coef


if __name__ == '__main__':
    # 生成的数据
    x, y, coef = create_dataset()
    # 绘制数据的真实的线性回归结果
    plt.scatter(x, y)
    x = torch.linspace(x.min(), x.max(), 1000)
    y1 = torch.tensor([v * coef + 14.5 for v in x])
    plt.plot(x, y1, label='真实结果')
    plt.grid()
    plt.legend()
    plt.show()
