'''
Description: 04反向传播算法BP    参考：https://blog.csdn.net/bit452/article/details/109643481
Author: HCQ
Company(School): UCAS
Email: 1756260160@qq.com
Date: 2020-11-30 17:49:44
LastEditTime: 2020-12-04 21:45:21
FilePath: /pytorch/PyTorch深度学习实践/04反向传播算法BP.py
'''


import torch
x_data = [1.0, 2.0, 3.0]
y_data = [2.0, 4.0, 6.0]
 
 # 定义w
w = torch.Tensor([1.0]) # w的初值为1.0， 别忘了中括号
w.requires_grad = True # 需要计算梯度
 
def forward(x):   # 被loss函数使用
    return x*w  # w是一个Tensor  x和w进行tensor和tensor之间的数乘 , x自动类型转换，转换为tensor
 
 
def loss(x, y):
    y_pred = forward(x)
    return (y_pred - y)**2
 
 # 训练过程
print("predict (before training)", 4, forward(4).item())  #  4 4.0
 
for epoch in range(100):
    for x, y in zip(x_data, y_data):
        l =loss(x,y) # l是一个张量，tensor主要是在建立计算图 forward, compute the loss
        l.backward() # 1. 自动计算所有梯度，然后把梯度存在各自有梯度的地方，本案例存放在w里面 backward,compute grad for Tensor whose requires_grad set to True
        # 2 只要backward，这个计算图就被释放了,下次再计算loss，会创建一个新的计算图
        print('\tgrad:', x, y, w.grad.item())   # w.grad = w的梯度
        print('\tgrad data:', x, y, w.grad.data)   # w.grad = w的梯度  w.grad.data :  tensor([-5.7220e-06])
        w.data = w.data - 0.01 * w.grad.data   # 权重更新时，需要用到标量，注意grad也是一个tensor，这里我们不是要计算图，而是数据相减
 
        w.grad.data.zero_() # after update, remember set the grad to zero  将梯度清零   在grad更新时，每一次运算后都需要将上一次的梯度的数据清空，
 
    print('progress:', epoch, l.item()) # 取出loss使用l.item，不要直接使用l（l是tensor会构建计算图）
 
print("predict (after training)", 4, forward(4).item())