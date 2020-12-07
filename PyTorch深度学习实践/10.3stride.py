'''
Description: 步长
Author: HCQ
Company(School): UCAS
Email: 1756260160@qq.com
Date: 2020-12-07 11:44:37
LastEditTime: 2020-12-07 11:45:20
FilePath: /pytorch/PyTorch深度学习实践/10.3stride.py
'''

import torch
input = [3,4,6,5,7,
         2,4,6,8,2,
         1,6,7,8,4,
         9,7,4,6,2,
         3,7,5,4,1]
input = torch.Tensor(input).view(1, 1, 5, 5)
#stride=2步长调整为2
conv_layer = torch.nn.Conv2d(1, 1, kernel_size=3, stride=2, bias=False)
kernel = torch.Tensor([1,2,3,4,5,6,7,8,9]).view(1, 1, 3, 3)
conv_layer.weight.data = kernel.data
output = conv_layer(input)
print(output.data)