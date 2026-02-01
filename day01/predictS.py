import torch


def dm01():
    # 点积运算
    data1 = torch.tensor([[1, 2], [3, 4], [5, 6]])
    data2 = torch.tensor([[5, 6], [7, 8]])

    # 方式一:
    data3 = data1 @ data2
    print("data3-->", data3)
    # 方式二:
    data4 = torch.matmul(data1, data2)
    print("data4-->", data4)


def dm02():
    # 设置随机种子
    torch.manual_seed(18)
    data = torch.randint(0, 10, [2, 3], dtype=torch.float64)
    print("data-->", data)
    # 1.计算均值
    # 注意:tensor必须为Float 或者 Double 类型
    print(data.mean())
    print(data.mean(dim=0))  # 列均值
    print(data.mean(dim=1))  # 行均值
    # 计算总和
    print(data.sum())
    print(data.sum(dim=0))
    print(data.sum(dim=1))
    # 计算平方
    print(torch.pow(data, 2))
    # 计算平方根
    print(data.sqrt())
    # 指数运算
    print(data.exp())
    # 对数计算
    print(data.log())  # e的幂次方
    print(data.log2())
    print(data.log10())


def dm03():
    torch.manual_seed(15858)
    data = torch.randint(0, 10, [4, 5])
    print("data-->", data)
    print(data[0])
    print(data[:, 0])
    # 返回(0,1),(1,2)两个位置的元素
    print(data[[0, 1], [1, 2]])
    # 返回0,1 行的1,2列共4个元素
    print(data[[[0], [1]], [1, 2]])
    # 前3行的前2列数据
    print(data[:3, :2])

    # 第二行到最后的前2列数据
    print(data[2:, :2])
    data = torch.randint(0, 10, [3, 4, 5])
    print("data-->", data)
    # 获取0轴上的第一个数据
    print(data[0, :, :])
    # 获取1轴上的第1个数据
    print(data[:, 0, :])
    # 获取2轴上的第1个数据
    print(data[:, :, 0])


def dm04():
    # data = torch.tensor([[10, 20, 30], [40, 50, 60]])
    # # 1.使用shape属性或者size方法都可以获得张量形状
    # print(data.shape, data.shape[0], data.shape[1])
    # print(data.size(), data.size(0), data.size(1))
    # # 使用reshape,函数修改张量形状
    # new_data = data.reshape(1, 6)
    # print(new_data.shape)

    mydata1 = torch.tensor([1,2,3,4,5])
    print('mydata1--->', mydata1.shape, mydata1)    # 一个普通的数组 1维数据
    mydata2 = mydata1.unsqueeze(dim = 0)
    print('在0维度上 拓展难度', mydata2.shape, mydata2)
    mydata3 = mydata1.unsqueeze(dim = 1)
    print('在1维度上 拓展难度', mydata3.shape, mydata3)
    mydata4 = mydata1.unsqueeze(dim = -1)
    print('在-1维度上 拓展难度', mydata4.shape, mydata4)
    mydata5 = mydata4.squeeze()
    print('压缩维度', mydata5.shape, mydata5)




if __name__ == '__main__':
    # dm01()
    # dm02()
    # dm03()
    dm04()
