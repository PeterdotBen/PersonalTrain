import torch

# 1,在指定区间按照步长生成元素[start, end, step]
data = torch.arange(0, 10, 2)
print(data)

# 2.在指定区间内按照元素个数生成[start, end, step]
data = torch.linspace(0, 10, 10)
print(data)