import torch


def dm01():
    torch.manual_seed(28)
    t1 = torch.randint(0, 10, (3, 4))
    print(f"t1:{t1}, shape:{t1.shape}, dtype:{t1.dtype}")

    # t2 = t1 + 1
    # # t2 = t1.add(1)
    # # t2 = torch.add(t1, 1)
    # # t2 = t1.add_(1)
    # print(f"t2:{t2}, shape:{t2.shape}, dtype:{t2.dtype}")
    # print(f"t1:{t1}, shape:{t1.shape}, dtype:{t1.dtype}")

    # t2 = t1 - 1
    # # t2 = t1.sub(1)
    # # t2 = torch.sub(t1, 1)
    # # t2 = t1.sub_(1)
    # print(f"t2:{t2}, shape:{t2.shape}, dtype:{t2.dtype}")
    # print(f"t1:{t1}, shape:{t1.shape}, dtype:{t1.dtype}")

    # t2 = t1 - 1
    # t2 = t1.neg(2)
    t2 = torch.neg(t1, 2)
    # # t2 = t1.sub_(1)
    print(f"t2:{t2}, shape:{t2.shape}, dtype:{t2.dtype}")
    print(f"t1:{t1}, shape:{t1.shape}, dtype:{t1.dtype}")




if __name__ == '__main__':
    dm01()
