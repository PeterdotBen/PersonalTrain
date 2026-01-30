import torch
import numpy as np

# 创建张量标量
data = torch.tensor(10)
print(data)
# 2.numpy 数组, 由于data为float64,下面代码也使用该类型
data = np.random.rand(2, 3)
data = torch.tensor(data)
print(data)
# 3.列表,下面代码使用默认元素 float32
data = [[10.0, 20.0, 30.0], [40.0, 50.0, 60.0]]
data = torch.tensor(data)
print(data)
