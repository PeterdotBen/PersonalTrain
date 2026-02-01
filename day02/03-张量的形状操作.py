import torch


def dm01():
    # 设置随机种子
    torch.manual_seed(28)
    # 创建一个张量(3,4)
    t1 = torch.randint(0, 10, (3, 4))
    print(f"t1:\n{t1}, shape:{t1.shape}, dtype:{t1.dtype}")
    # 2.使用reshape()方法将张量形状改为(4,3)
    t2 = t1.reshape(-1, 2)
    print(f"t2:\n{t2}, shape:{t2.shape}, dtype:{t2.dtype}")
    # 3.unsqueeze()方法增加一个维度
    t3 = t1.unsqueeze(dim=0).unsqueeze(0)
    print(f"t3:\n{t3}, shape:{t3.shape}, dtype:{t3.dtype}")
    # 4.squeeze()方法删除一个维度
    t4 = t3.squeeze(dim=0).squeeze(0)
    print(f"t4:\n{t4}, shape:{t4.shape}, dtype:{t4.dtype}")


def dm02():
    torch.manual_seed(28)
    t1 = torch.randint(0, 10, (3, 4, 5))
    print(f"t1:\n{t1}, shape:{t1.shape}, dtype:{t1.dtype}")
    t2 = t1.permute(2,0,1)
    print(f"t2:\n{t2}, shape:{t2.shape}, dtype:{t2.dtype}")
    print(f't2是否是连续张量:{t2.is_contiguous()}')

    # 使用transpose()方法
    t3 = t1.transpose(2,1)
    print(f"t3:\n{t3}, shape:{t3.shape}, dtype:{t3.dtype}")
    # 使用view()方法
    t4 = t1.view(5,3,4)
    print(f"t4:\n{t4}, shape:{t4.shape}, dtype:{t4.dtype}")
    t4 = t2.contiguous().view(3,4,5)
    print(f"t4:\n{t4}, shape:{t4.shape}, dtype:{t4.dtype}")


if __name__ == '__main__':
    dm02()
