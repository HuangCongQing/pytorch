'''
Description: downsampling
Author: HCQ
Company(School): UCAS
Email: 1756260160@qq.com
Date: 2020-12-07 11:46:39
LastEditTime: 2020-12-07 11:47:15
FilePath: /pytorch/PyTorch深度学习实践/10.4downsampling.py
'''
import torch
input = [3,4,6,5,
         2,4,6,8,
         1,6,7,8,
         9,7,4,6, ]
input = torch.Tensor(input).view(1, 1, 4, 4)
#创建一个2x2的最大池化层
maxpooling_layer = torch.nn.MaxPool2d(kernel_size=2)
output = maxpooling_layer(input)
print(output)