import torch

def dm01():
    t1 = torch.randint(0, 10, (3, 4, 5))
    print(f"t1:{t1}, shape:{t1.shape}, dtype:{t1.dtype}")

    # 演示sum ()mean(),max(),min()
    # t2 = t1.sum()
    # print(f"t2:{t2}, shape:{t2.shape}, dtype:{t2.dtype}")
    # t2 = t1.to(torch.float32)
    # t2 = t2.mean()
    # print(f"t2:{t2}, shape:{t2.shape}, dtype:{t2.dtype}")
    # t2 = t1.max()
    # print(f"t2:{t2}, shape:{t2.shape}, dtype:{t2.dtype}")
    t2 = t1.min()
    print(f"t2:{t2}, shape:{t2.shape}, dtype:{t2.dtype}")
    print(f"t1.min(dim=0):{t1.min(dim=0)}, shape:{t1.min(dim=0)[0].shape}")
    print(f"t1.min(dim=0):{t1.min(dim=1)}, shape:{t1.min(dim=1)[0].shape}")


if __name__ == '__main__':
    dm01()