import torch


def dm01():
    torch.manual_seed(28)
    t1 = torch.randn(3, 4)
    print(f"t1:{t1}, shape:{t1.shape}, dtype:{t1.dtype}")

    n2 = t1.numpy().copy()
    print(f"n2:{n2}, shape:{n2.shape}, dtype:{n2.dtype}")

    n2[0, 0] = 200
    print(f"n2:{n2}, shape:{n2.shape}, dtype:{n2.dtype}")
    print(f"t1:{t1}, shape:{t1.shape}, dtype:{t1.dtype}")

    t2 = torch.from_numpy(n2)
    print(f"t2:{t2}, shape:{t2.shape}, dtype:{t2.dtype}")
    t2[0, 0] = 10
    print(f"t2:{t2}, shape:{t2.shape}, dtype:{t2.dtype}")
    print(f"n2:{n2}, shape:{n2.shape}, dtype:{n2.dtype}")

    t3 = torch.tensor(n2)
    print(f"t3:{t3}, shape:{t3.shape}, dtype:{t3.dtype}")
    t3[0, 0] = 20
    print(f"t3:{t3}, shape:{t3.shape}, dtype:{t3.dtype}")
    print(f"n2:{n2}, shape:{n2.shape}, dtype:{n2.dtype}")

    t4 = torch.tensor(True)
    print(f"t4:{t4}, shape:{t4.shape}, dtype:{t4.dtype}")
    a4 = t4.item()
    print(f"a4:{a4}, type:{type(a4)}")


if __name__ == '__main__':
    dm01()
