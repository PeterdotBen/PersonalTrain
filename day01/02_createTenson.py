# 1.创建2行3列的张量, 默认dtype为float32
import torch
torch.set_printoptions(precision=10)

data = torch.Tensor(2, 3)
print(data)
# 2.注意:如果传递列表,则创建包含指定元素的张量
data = torch.Tensor([10])
print(data)
data = torch.Tensor([[10]])
print(data)
data = torch.Tensor([10, 20])
print(data)