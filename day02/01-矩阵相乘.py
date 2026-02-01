import torch

def dm01():
    torch.manual_seed(28)
    t1 = torch.randint(0, 10, (3, 4))
    print(f"t1:{t1}, shape:{t1.shape}, dtype:{t1.dtype}")
    t2 = torch.randint(0, 10, (3, 4))
    print(f"t2:{t2}, shape:{t2.shape}, dtype:{t2.dtype}")

    t3  =torch.mul(t1, t2)
    print(f"t3:{t3}, shape:{t3.shape}, dtype:{t3.dtype}")

def dm02():
    torch.manual_seed(28)
    t1 = torch.randint(0, 10, (3, 4))
    print(f"t1:{t1}, shape:{t1.ndim}, dtype:{t1.dtype}")
    t2 = torch.randint(0, 10, (4, 3))
    print(f"t2:{t2}, shape:{t2.shape}, dtype:{t2.dtype}")
    t3 = torch.matmul(t1, t2)
    print(f"t3:{t3}, shape:{t3.shape}, dtype:{t3.dtype}")




if __name__ == '__main__':
    dm02()