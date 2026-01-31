import torch
data = torch.randn(2, 3)
print(data)

# 查看随机种子
print('随机种子:', torch.initial_seed())

# 2.随机数种子设置
torch.manual_seed(100)
data = torch.randn(2, 3)
print(data)
print('随机种子:', torch.initial_seed())