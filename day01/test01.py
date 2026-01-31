import torch
print(torch.__version__)              # 应该是 2.10.0 或更高
print(torch.version.cuda)             # 应该是 12.8
print(torch.cuda.is_available())      # 应该返回 True
print(torch.cuda.get_device_name(0))  # 显示你的 GPU 名称