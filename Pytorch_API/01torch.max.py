'''
Description: https://blog.csdn.net/weixin_43255962/article/details/84402586?
Author: HCQ
Company(School): UCAS
Email: 1756260160@qq.com
Date: 2021-06-16 16:19:46
LastEditTime: 2021-06-16 16:23:47
FilePath: /pytorch/Pytorch_API/01torch.max.py
'''
import torch

x = torch.rand(4,4)
print('x:\n',x)
print('torch.max(x,1):\n',torch.max(x,1))
print('torch.max(x,0):\n',torch.max(x,0))
print('torch.max(x,1)[0]:\n',torch.max(x,1)[0])
print('torch.max(x,1)[1]:\n',torch.max(x,1)[1])
print('torch.max(x,1)[1].data:\n',torch.max(x,1)[1].data)
print('torch.max(x,1)[1].data.numpy():\n',torch.max(x,1)[1].data.numpy())
print('torch.max(x,1)[1].data.numpy().squeeze():\n',torch.max(x,1)[1].data.numpy().squeeze())
print('torch.max(x,1)[0].data:\n',torch.max(x,1)[0].data)
print('torch.max(x,1)[0].data.numpy():\n',torch.max(x,1)[0].data.numpy())
print('torch.max(x,1)[0].data.numpy().squeeze():\n',torch.max(x,1)[0].data.numpy().squeeze())
''' 
x:
 tensor([[0.8879, 0.3753, 0.4216, 0.9018],
        [0.9666, 0.2836, 0.8554, 0.1083],
        [0.8416, 0.2497, 0.9407, 0.0587],
        [0.4886, 0.0165, 0.4031, 0.0805]])

torch.max(x,1):
 torch.return_types.max(
values=tensor([0.9018, 0.9666, 0.9407, 0.4886]),
indices=tensor([3, 0, 2, 0]))

torch.max(x,0):
 torch.return_types.max(
values=tensor([0.9666, 0.3753, 0.9407, 0.9018]),
indices=tensor([1, 0, 2, 0]))

torch.max(x,1)[0]: # 数组
 tensor([0.9018, 0.9666, 0.9407, 0.4886])

torch.max(x,1)[1]: # 下标
 tensor([3, 0, 2, 0])
 
torch.max(x,1)[1].data:
 tensor([3, 0, 2, 0])
torch.max(x,1)[1].data.numpy():
 [3 0 2 0]
torch.max(x,1)[1].data.numpy().squeeze():
 [3 0 2 0]
torch.max(x,1)[0].data:
 tensor([0.9018, 0.9666, 0.9407, 0.4886])
torch.max(x,1)[0].data.numpy():
 [0.9018364  0.9665782  0.9406531  0.48861176]
torch.max(x,1)[0].data.numpy().squeeze():
 [0.9018364  0.9665782  0.9406531  0.48861176]


 '''