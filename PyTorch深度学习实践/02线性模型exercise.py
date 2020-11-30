'''
Description: 实现线性模型（y=wx+b）并输出loss的3D图像。 参考:https://blog.csdn.net/weixin_44841652/article/details/105017087
Author: HCQ
Company(School): UCAS
Email: 1756260160@qq.com
Date: 2020-11-30 15:02:52
LastEditTime: 2020-11-30 15:31:51
FilePath: /pytorch/PyTorch深度学习实践/02线性模型exercise.py
'''
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D # 三维图

#这里设函数为y=3x+2
x_data = [1.0,2.0,3.0]
y_data = [5.0,8.0,11.0]

def forward(x):
    return x * w + b  # 计算

def loss(x,y):
    y_pred = forward(x)
    return (y_pred-y)*(y_pred-y)

mse_list = []
W=np.arange(0.0,4.1,0.1)
B=np.arange(0.0,4.1,0.1)
[w,b]=np.meshgrid(W,B) # 一句话解释numpy.meshgrid()——生成网格点坐标矩阵。
# print([w,b])

l_sum = 0
for x_val, y_val in zip(x_data, y_data):
    y_pred_val = forward(x_val) # 预测y值
    print(y_pred_val)
    loss_val = loss(x_val, y_val)
    l_sum += loss_val

fig = plt.figure()
ax = Axes3D(fig)
ax.plot_surface(w, b, l_sum/3) # 可视化
plt.show()
