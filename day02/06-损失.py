import torch
#3.使用torch.randint()创建一个形状为(3,4)的随机整数张量,求所有元素的均值 和 第1个轴的均值，并打印。(注意元素类型转换)
t1 = torch.randint(0, 10, (3, 4)).to(torch.float32)
print(f"t1:{t1}, shape:{t1.shape}, dtype:{t1.dtype}")
print(f"t1.mean():{t1.mean()}, t1.mean(dim=0):{t1.mean(dim=0)}")
