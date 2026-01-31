import torch


def dm01():
    t1 = torch.arange(0, 10)
    print(f"1d-t1:{t1}, shape:{t1.shape},type:{type(t1)}, dtype:{t1.dtype}")
    t2 = torch.linspace(0, 9, 10)
    print(f"1d-t2:{t2}, shape:{t2.shape},type:{type(t2)}, dtype:{t2.dtype}")


def dm02():
    torch.manual_seed(28)
    t1 = torch.rand(3, 5)
    print(f"2d-t1:{t1}, shape:{t1.shape},type:{type(t1)}, dtype:{t1.dtype}")
    t2 = torch.randn(3, 5)
    print(f"2d-t2:{t2}, shape:{t2.shape},type:{type(t2)}, dtype:{t2.dtype}")
    t3 = torch.randint(0, 10, (3, 5))
    print(f"2d-t3:{t3}, shape:{t3.shape},type:{type(t3)}, dtype:{t3.dtype}")


if __name__ == '__main__':
    dm01()
    dm02()
