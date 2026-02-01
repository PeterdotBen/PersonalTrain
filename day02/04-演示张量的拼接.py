import torch


def dm01():
    torch.manual_seed(28)
    # 1.定义两个张量(3,4),(5,4)
    t1 = torch.randint(0, 10, (3, 4))
    t2 = torch.randint(0, 10, (3, 4))
    # t4 = torch.randint(0, 10, (5, 4))
    print(f"t1:\n{t1}, shape:{t1.shape}, dtype:{t1.dtype}")
    print(f"t2:\n{t2}, shape:{t2.shape}, dtype:{t2.dtype}")
    # 2.使用torch.cat()方法拼接t1,t2
    # t3 = torch.cat([t1, t2], dim=0)
    # print(f"t3:{t3}, shape:{t3.shape}, dtype:{t3.dtype}")
    # t3 = torch.cat([t1, t2], dim=1)
    # print(f"t3:{t3}, shape:{t3.shape}, dtype:{t3.dtype}")
    # # 3.使用torch.stack()方法拼接t1,t2
    t4= torch.stack([t1, t2], dim=0)
    print(f"t4:\n{t4}, shape:{t4.shape}, dtype:{t4.dtype}")         # (2,3,4)
    t4 = torch.stack([t1, t2], dim=1)
    print(f"t4:\n{t4}, shape:{t4.shape}, dtype:{t4.dtype}")          # (3,2,4)
    t4 = torch.stack([t1, t2], dim=2)
    print(f"t4:\n{t4}, shape:{t4.shape}, dtype:{t4.dtype}")          # (3,4,2)
    ...


if __name__ == '__main__':
    dm01()