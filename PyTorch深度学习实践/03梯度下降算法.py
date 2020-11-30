'''
Description: 梯度下降
Author: HCQ
Company(School): UCAS
Email: 1756260160@qq.com
Date: 2020-11-30 15:50:54
LastEditTime: 2020-11-30 17:19:25
FilePath: /pytorch/PyTorch深度学习实践/03梯度下降算法.py
'''
import matplotlib.pyplot as plt
 
# prepare the training set
x_data = [1.0, 2.0, 3.0]
y_data = [2.0, 4.0, 6.0]
 
# initial guess of weight 
w = 1.0  # 初始化权重
 
# define the model linear model y = w*x
def forward(x):
    return x*w
 
#define the cost function MSE 
def cost(xs, ys):
    cost = 0
    for x, y in zip(xs,ys):
        y_pred = forward(x)
        cost += (y_pred - y)**2  # 
    return cost / len(xs)
 
# define the gradient function  gd 梯度
def gradient(xs,ys):
    grad = 0
    for x, y in zip(xs,ys):
        grad += 2*x*(x*w - y) # ∂cost/ ∂w,对w求导
    return grad / len(xs)
 
epoch_list = []
cost_list = []
print('predict (before training)', 4, forward(4))
for epoch in range(100):
    cost_val = cost(x_data, y_data)
    grad_val = gradient(x_data, y_data)# 1 求梯度
    w-= 0.01 * gradw_val  # 0.01 learning rate # 2  更新斜率（权重）
    print('epoch:', epoch, 'w=', w, 'loss=', cost_val)
    epoch_list.append(epoch)
    cost_list.append(cost_val)
 
print('predict (after training)', 4, forward(4))
plt.plot(epoch_list,cost_list)
plt.ylabel('cost')
plt.xlabel('epoch')
plt.show() 