'''
Description: 线性模型
Author: HCQ
Company(School): UCAS
Email: 1756260160@qq.com
Date: 2020-11-30 14:36:45
LastEditTime: 2020-11-30 14:38:56
FilePath: /pytorch/PyTorch深度学习实践/02线性模型.py
'''
import numpy as np
import matplotlib.pyplot as plt
 
x_data = [1.0, 2.0, 3.0]
y_data = [2.0, 4.0, 6.0]
 
 
def forward(x):
    return x*w
 
 
def loss(x, y):
    y_pred = forward(x)
    return (y_pred - y)**2
 
 
# 穷举法
w_list = []
mse_list = []
for w in np.arange(0.0, 4.1, 0.1):
    print("w=", w) # 输出w
    l_sum = 0
    for x_val, y_val in zip(x_data, y_data):  # zip拼接数据
        y_pred_val = forward(x_val)  # 预测的y值
        loss_val = loss(x_val, y_val)
        l_sum += loss_val   # loss求和
        print('\t', x_val, y_val, y_pred_val, loss_val)
    print('MSE=', l_sum/3)  # 输出mse
    w_list.append(w)
    mse_list.append(l_sum/3)
    
plt.plot(w_list,mse_list)
plt.ylabel('Loss')
plt.xlabel('w')
plt.show()    