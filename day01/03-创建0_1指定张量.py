import torch

def dm01():
    t1 =torch.zeros(3,4)
    print(f"2d-t1:{t1}, shape:{t1.shape},type:{type(t1)}, dtype:{t1.dtype}")
    t2 = torch.randn(3,5)
    print(f"2d-t2:{t2}, shape:{t2.shape},type:{type(t2)}, dtype:{t2.dtype}")
    t3 = torch.zeros_like(t2)
    print(f"2d-t3:{t3}, shape:{t3.shape},type:{type(t3)}, dtype:{t3.dtype}")

def dm02():
    t1 =torch.ones(3,4)
    print(f"2d-t1:{t1}, shape:{t1.shape},type:{type(t1)}, dtype:{t1.dtype}")
    t2 = torch.randn(3,5)
    print(f"2d-t2:{t2}, shape:{t2.shape},type:{type(t2)}, dtype:{t2.dtype}")
    t3 = torch.ones_like(t2)
    print(f"2d-t3:{t3}, shape:{t3.shape},type:{type(t3)}, dtype:{t3.dtype}")


def dm03():
    t1 =torch.full((3,4),10)
    print(f"2d-t1:{t1}, shape:{t1.shape},type:{type(t1)}, dtype:{t1.dtype}")
    t2 = torch.randn(3,5)
    print(f"2d-t2:{t2}, shape:{t2.shape},type:{type(t2)}, dtype:{t2.dtype}")
    t3 = torch.full_like(t2,6)
    print(f"2d-t3:{t3}, shape:{t3.shape},type:{type(t3)}, dtype:{t3.dtype}")

if __name__ == '__main__':
    dm01()
    dm02()
    dm03()