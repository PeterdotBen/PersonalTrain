import torch
def dm01():
    t1 = torch.tensor(1)
    print(f"0d-t1:{t1}, shape:{t1.shape},type:{type(t1)}, dtype:{t1.dtype}")
    # 2. 创建1维张量
    t2 = torch.tensor(1)
    print(f"0d-t2:{t2}, shape:{t2.shape},type:{type(t2)}, dtype:{t2.dtype}")
    t3 = torch.tensor(1)
    print(f"0d-t3:{t3}, shape:{t3.shape},type:{type(t3)}, dtype:{t3.dtype}")
    t4 = torch.tensor(1)
    print(f"0d-t4:{t4}, shape:{t4.shape},type:{type(t4)}, dtype:{t4.dtype}")


def dm02():
    t1 = torch.Tensor(1)
    print(f"0d-t1:{t1}, shape:{t1.shape},type:{type(t1)}, dtype:{t1.dtype}")
    # 2. 创建1维张量
    t2 = torch.Tensor(1)
    print(f"0d-t2:{t2}, shape:{t2.shape},type:{type(t2)}, dtype:{t2.dtype}")
    t3 = torch.Tensor(1)
    print(f"0d-t3:{t3}, shape:{t3.shape},type:{type(t3)}, dtype:{t3.dtype}")
    t4 = torch.Tensor(1)
    print(f"0d-t4:{t4}, shape:{t4.shape},type:{type(t4)}, dtype:{t4.dtype}")

def dm03():
    def dm01():
        t1 = torch.tensor(1)
        print(f"0d-t1:{t1}, shape:{t1.shape},type:{type(t1)}, dtype:{t1.dtype}")
        # 2. 创建1维张量
        t2 = torch.tensor(1)
        print(f"0d-t2:{t2}, shape:{t2.shape},type:{type(t2)}, dtype:{t2.dtype}")
        t3 = torch.tensor(1)
        print(f"0d-t3:{t3}, shape:{t3.shape},type:{type(t3)}, dtype:{t3.dtype}")
        t4 = torch.tensor(1)
        print(f"0d-t4:{t4}, shape:{t4.shape},type:{type(t4)}, dtype:{t4.dtype}")
def dm03():
    # 1.创建0维张量/0D张量，也就是标量
    t1 = torch.IntTensor(1)
    print(f"0D-t1:{t1}, shape: {t1.shape}, type: {type(t1)}, dtype: {t1.dtype}")
    # 2.创建1维张量/1D张量
    t2 = torch.IntTensor([1,2,3])
    print(f"1D-t2:{t2}, shape: {t2.shape}, type: {type(t2)}, dtype: {t2.dtype}")
    # 3.创建2维张量/2D张量
    t3 = torch.FloatTensor([[1,2,3],[4,5,6]])    # 形状(2,3)
    print(f"2D-t3:{t3}, shape: {t3.shape}, type: {type(t3)}, dtype: {t3.dtype}")
    # 4.创建3维张量/3D张量
    t4 = torch.DoubleTensor([[[1,2,3],[4,5,6]]])  # 形状(1,2,3)
    print(f"3D-t4:{t4}, shape: {t4.shape}, type: {type(t4)}, dtype: {t4.dtype}")
    # 5.创建指定形状的张量，(3,4)
    t5 = torch.Tensor(3,4)
    print(f"指定形状-t5:{t5}, shape: {t5.shape}")

if __name__ == '__main__':
    # dm01()
    # dm02()
    dm03()