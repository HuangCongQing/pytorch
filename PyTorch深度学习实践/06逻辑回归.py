'''
Description: 逻辑回归
视频：https://www.bilibili.com/video/BV1Y7411d7Ys?p=6
博客： https://blog.csdn.net/bit452/article/details/109680909
Author: HCQ
Company(School): UCAS
Email: 1756260160@qq.com
Date: 2020-12-05 16:53:13
LastEditTime: 2020-12-05 17:12:56
FilePath: /pytorch/PyTorch深度学习实践/06逻辑回归.py
'''


import torch
# import torch.nn.functional as F
 
# 1 prepare dataset
x_data = torch.Tensor([[1.0], [2.0], [3.0]])
y_data = torch.Tensor([[0], [0], [1]]) # =============================[0], [0], [1]]=========================
 
# 2 design model using class
class LogisticRegressionModel(torch.nn.Module):
    def __init__(self):
        super(LogisticRegressionModel, self).__init__()
        self.linear = torch.nn.Linear(1,1)
 
    def forward(self, x):
        # y_pred = F.sigmoid(self.linear(x))
        y_pred = torch.sigmoid(self.linear(x)) # ========================sigmoid=============================
        return y_pred
model = LogisticRegressionModel()
 
# 3 construct loss and optimizer
# 默认情况下，loss会基于element平均，如果size_average=False的话，loss会被累加。
criterion = torch.nn.BCELoss(size_average = False)  # ==================BCELoss===========================
optimizer = torch.optim.SGD(model.parameters(), lr = 0.01)
 
# 4 training cycle forward, backward, update
for epoch in range(1000):
    y_pred = model(x_data)
    loss = criterion(y_pred, y_data)
    print(epoch, loss.item())
 
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
 
print('w = ', model.linear.weight.item())
print('b = ', model.linear.bias.item())
 
x_test = torch.Tensor([[4.0]])
y_test = model(x_test)
print('y_pred = ', y_test.data)



# 可视化

import numpy as np
import matplotlib.pyplot as plt
x = np.linspace(0, 10, 200)
x_t = torch.Tensor(x).view((200, 1))
y_t = model(x_t)  # 使用训练好的模型
y = y_t.data.numpy()
plt.plot(x, y)
plt.plot([0, 10], [0.5, 0.5], c='r')
plt.xlabel('Hours')
plt.ylabel('Probability of Pass')
plt.grid()
plt.show()



