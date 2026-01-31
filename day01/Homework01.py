import torch


def dm01():
    # 使用torch.tensor(),torch.arange(),randn()创建一个形状(3,4)的张量，并打印张量
    t1 = torch.tensor([[1, 2, 3, 4], [4, 5, 6, 7], [8, 9, 10, 11]])
    print(f"t1:{t1}, shape:{t1.shape}, dtype:{t1.dtype}")
    t2 = torch.arange(12).reshape(3, 4)
    print(f"t2:{t2}, shape:{t2.shape}, dtype:{t2.dtype}")
    t3 = torch.randn(3, 4)
    print(f"t3:{t3}, shape:{t3.shape}, dtype:{t3.dtype}")


def dm02():
    t1 = torch.zeros(5, 10)
    print(f"t1:{t1}, shape:{t1.shape}, dtype:{t1.dtype}")
    t2 = torch.ones(5, 10)
    print(f"t2:{t2}, shape:{t2.shape}, dtype:{t2.dtype}")


def dm03():
    t1 = torch.randint(0, 10, (3, 4)).to(torch.float32)
    print(f"t1:{t1}, shape:{t1.shape}, dtype:{t1.dtype}")


def dm04():
    t1 = torch.rand(3, 4).numpy()
    t1[1, 3] = 42
    t1= torch.from_numpy(t1)
    print(f"t1:{t1}, shape:{t1.shape}, dtype:{t1.dtype}")

def dm05():
    # 创建一个标量张量, 并转换为python数值,并打印
    t1 = torch.tensor(4)
    t2 = t1.item()
    print(f"t2:{t2}, type:{type(t2)}")


if __name__ == '__main__':
    dm01()
    dm02()
    dm03()
    dm04()
    dm05()
