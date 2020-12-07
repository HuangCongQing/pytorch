'''
Description: padding
Author: HCQ
Company(School): UCAS
Email: 1756260160@qq.com
Date: 2020-12-07 11:39:36
LastEditTime: 2020-12-07 11:40:23
FilePath: /pytorch/PyTorch深度学习实践/10.2padding.py
'''

import torch
input = [3,4,6,5,7,
         2,4,6,8,2,
         1,6,7,8,4,
         9,7,4,6,2,
         3,7,5,4,1]
#将一个列表转换成一个batch1,通道1，长宽5的张量
input = torch.Tensor(input).view(1, 1, 5, 5)
#卷积层padding=1也就是在外面加一圈
conv_layer = torch.nn.Conv2d(1, 1, kernel_size=3, padding=1, bias=False)
#定义一个卷积核
kernel = torch.Tensor([1,2,3,4,5,6,7,8,9]).view(1, 1, 3, 3)
#我们将自己设置的卷积核权重设置给卷积层的权重
conv_layer.weight.data = kernel.data
output = conv_layer(input)
print(output.data)