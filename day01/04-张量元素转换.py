import torch

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"device is {device}")


def dm01():
    t1 = torch.tensor([[1, 2, 3], [4, 5, 6]], dtype=torch.float32)
    print(f"t1:{t1}, shape:{t1.shape},type:{type(t1)}, dtype:{t1.dtype}")
    t2 = t1.to(dtype=torch.float64)
    print(f"t2:{t2}, shape:{t2.shape},type:{type(t2)}, dtype:{t2.dtype}")
    t3 = t2.to(device)
    print(f"t3:{t3}, shape:{t3.shape},type:{type(t3)}, dtype:{t3.dtype}")
    t3 = t1.type(dtype=torch.float64)
    print(f"t3:{t3}, shape:{t3.shape},type:{type(t3)}, dtype:{t3.dtype}")
    t4 = t1.double()
    print(f"t4:{t4}, shape:{t4.shape},type:{type(t4)}, dtype:{t4.dtype}")


if __name__ == '__main__':
    dm01()
