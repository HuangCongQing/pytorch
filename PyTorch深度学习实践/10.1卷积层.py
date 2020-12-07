'''
Description: 卷积层
Author: HCQ
Company(School): UCAS
Email: 1756260160@qq.com
Date: 2020-12-07 11:20:03
LastEditTime: 2020-12-07 11:20:26
FilePath: /pytorch/PyTorch深度学习实践/10.1卷积层.py
'''



import torch

in_channels, out_channels= 5, 10 #输入和输出的通道数
width, height = 100, 100 #图像大小
kernel_size = 3 #卷积核大小
batch_size = 1

#在torch中输入的数据都是小批量的，所以在输入的时候需要指定，一组有多少个张量
#torch.randn（）的作用是标准正态分布中随机取数，返回一个满足输入的batch，通道数，宽，高的大小的张量
input = torch.randn(batch_size,in_channels,width, height)
#torch.nn.Conv2d（输入通道数量，输出通道数量，卷积核大小）创建一个卷积对象
conv_layer = torch.nn.Conv2d(in_channels,out_channels,kernel_size=kernel_size)
output = conv_layer(input)

print(input.shape)
print(output.shape)
print(conv_layer.weight.shape)
