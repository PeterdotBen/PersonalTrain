import torch




# 1.创建指定形状全0张量
data = torch.zeros(2, 3)
print(data)

# 2.创建指定形状全0张量
data = torch.zeros_like(data)
print(data)
# 3.创建指定形状全1张量
data = torch.ones(2, 3)
print(data)
# 4.创建指定形状全1张量
data = torch.ones_like(data)
print(data)
data = torch.full((2, 3), 10)
print(data.type())