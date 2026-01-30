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

# 1.创建2行3列, dtype 为 int32的张量
data = torch.IntTensor(2, 3)
print(data)
# 2.注意:如果传递列表,则创建包含指定元素的张量
data = torch.IntTensor([2.5,3.3])
print(data)
# 其他类型
data = torch.LongTensor()
data2 = torch.ShortTensor()
data3 = torch.FloatTensor()
data4 = torch.DoubleTensor()
print(data, data2, data3, data4)