'''
Description: 03_1随机梯度下降算法  参考：https://blog.csdn.net/bit452/article/details/109637108
Author: HCQ
Company(School): UCAS
Email: 1756260160@qq.com
Date: 2020-11-30 17:39:01
LastEditTime: 2020-11-30 17:43:05
FilePath: /pytorch/PyTorch深度学习实践/03_1随机梯度下降算法.py
'''

''' 
随机梯度下降法在神经网络中被证明是有效的。效率较低(时间复杂度较高)，学习性能较好。

随机梯度下降法和梯度下降法的主要区别在于：

1、损失函数由cost()更改为loss()。cost是计算所有训练数据的损失，loss是计算一个训练函数的损失。对应于源代码则是少了两个for循环。

2、梯度函数gradient()由计算所有训练数据的梯度更改为计算一个训练数据的梯度。

3、本算法中的随机梯度主要是指，每次拿一个训练数据来训练，然后更新梯度参数。本算法中梯度总共更新100(epoch)x3 = 300次。梯度下降法中梯度总共更新100(epoch)次。
 '''

import matplotlib.pyplot as plt
 
x_data = [1.0, 2.0, 3.0]
y_data = [2.0, 4.0, 6.0]
 
w = 1.0
 
def forward(x):
    return x*w
 
# calculate loss function
def loss(x, y):
    y_pred = forward(x)
    return (y_pred - y)**2
 
# define the gradient function  sgd
def gradient(x, y):
    return 2*x*(x*w - y)
 
epoch_list = []
loss_list = []
print('predict (before training)', 4, forward(4))
for epoch in range(100):
    for x,y in zip(x_data, y_data):
        grad = gradient(x,y)
        w = w - 0.01*grad    # update weight by every grad of sample of training set   没法并行运算
        print("\tgrad:", x, y,grad)
        l = loss(x,y)
    print("progress:",epoch,"w=",w,"loss=",l)
    epoch_list.append(epoch)
    loss_list.append(l)
 
print('predict (after training)', 4, forward(4))
plt.plot(epoch_list,loss_list)
plt.ylabel('loss')
plt.xlabel('epoch')
plt.show() 