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
    # plt.scatter(x, y)
    # x = torch.linspace(x.min(), x.max(), 1000)
    # y1 = torch.tensor([v * coef + 14.5 for v in x])
    # plt.plot(x, y1, label='真实结果')
    # plt.grid()
    # plt.legend()
    # plt.show()
    # 构造数据集对象
    dataset = TensorDataset(x, y)
    # 创建数据加载器
    # dataset= :数据集对象
    # batch_size=: 批量训练样本数据
    # shuffle=: 是否打乱数据集
    dataloader = DataLoader(dataset, batch_size=10, shuffle=True)
    # 构造模型
    # in_features指的是输入张量大小size
    # out_features指的是输出张量大小size
    model = nn.Linear(in_features=1, out_features=1)
    # 构造平方损失函数
    loss_fn = nn.MSELoss()
    # 构造优化器
    optimizer = optim.SGD(model.parameters(), lr=1e-2)
    # 定义训练轮数
    epochs = 100
    # 定义没练的平均损失列表
    train_loss_list = []
    # 开始训练
    for epoch in range(epochs):
        # 初始化 训练总损失数
        total_train_loss = 0
        total_train_sample = 0
        for train_x, train_y in dataloader:
            # 模型预测,向前传播
            y_pred = model(train_x)
            # 计算损失
            loss = loss_fn(y_pred, train_y.reshape(-1, 1))
            # 计算总损失 和总样本数
            batch_size = train_x.size(0)
            total_train_loss += loss.item() * batch_size
            total_train_sample += batch_size
            # 梯度清零
            optimizer.zero_grad()
            # 反向传播,计算梯度
            loss.backward()
            # 更新参数
            optimizer.step()
        # 计算平均损失
        train_loss_list.append(total_train_loss / total_train_sample)
        print(f"训练轮数:{epoch + 1}, 平均损失:{train_loss_list[-1]}")

    # 绘制训练损失
    plt.plot(range(epochs), train_loss_list)
    plt.title("损失变化曲线")
    plt.grid()
    plt.show()
    # 绘制拟合直线
    plt.scatter(x, y)
    x = torch.linspace(x.min(), x.max(), 1000)
    y1 = torch.tensor([v * model.weight + model.bias for v in x])
    y2 = torch.tensor([v * coef + 14.5 for v in x])
    plt.plot(x, y1, label='训练结果')
    plt.plot(x, y2, label='真实结果')
    plt.grid()
    plt.legend()
    plt.show()
