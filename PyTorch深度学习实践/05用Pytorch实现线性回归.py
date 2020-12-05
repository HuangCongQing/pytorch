'''
Description: 用Pytorch实现线性回归
https://blog.csdn.net/bit452/article/details/109677086
https://blog.csdn.net/weixin_44841652/article/details/105068509
Author: HCQ
Company(School): UCAS
Email: 1756260160@qq.com
Date: 2020-12-05 12:20:13
LastEditTime: 2020-12-05 16:54:16
FilePath: /pytorch/PyTorch深度学习实践/05用Pytorch实现线性回归.py
'''


import torch
# prepare dataset
# x,y是矩阵，3行1列 也就是说总共有3个数据，每个数据只有1个特征
x_data = torch.Tensor([[1.0], [2.0], [3.0]])
y_data = torch.Tensor([[2.0], [4.0], [6.0]])
 
#design model using class
"""
our model class should be inherit from nn.Module, which is base class for all neural network modules.
member methods __init__() and forward() have to be implemented
class nn.linear contain two member Tensors: weight and bias
class nn.Linear has implemented the magic method __call__(),which enable the instance of the class can
be called just like a function.Normally the forward() will be called 
"""
class LinearModel(torch.nn.Module):  # torch.nn.Module是父类  
    def __init__(self):  # 构造函数(初始化)
        super(LinearModel, self).__init__() # 调用父类的init     super(LinearModel, self)  == torch.nn.Module
        # (1,1)是指输入x和输出y的特征维度，这里数据集中的x和y的特征都是1维的
        # 该线性层需要学习的参数是w和b  获取w/b的方式分别是~linear.weight/linear.bias
        self.linear = torch.nn.Linear(1, 1) # 构造对象，执行wx+b的操作，Linear也是继承自Model，并说明输入输出的维数，第三个参数默认为true，表示用到b
 
    def forward(self, x):
        y_pred = self.linear(x) #可调用对象（对象括号里面加参数x），计算y=wx+b 
        return y_pred
 
model = LinearModel() # 自动构建计算图，实现反向（自带的，不用写）
 
# construct loss and optimizer
# criterion = torch.nn.MSELoss(size_average = False)
criterion = torch.nn.MSELoss(reduction = 'sum')
optimizer = torch.optim.SGD(model.parameters(), lr = 0.01) # model.parameters()自动完成所有设计权重参数的初始化操作
 
# training cycle forward, backward, update
for epoch in range(100):
    y_pred = model(x_data) # forward:predict    等同于self.linear(x_data）
    loss = criterion(y_pred, y_data) # forward: loss
    print(epoch, loss.item())
 
    optimizer.zero_grad() # the grad computer by .backward() will be accumulated. so before backward, remember set the grad to zero
    loss.backward() # backward: autograd
    optimizer.step() # update 参数，即更新w和b的值
 
print('w = ', model.linear.weight.item())
print('b = ', model.linear.bias.item())
 
x_test = torch.Tensor([[4.0]])
y_test = model(x_test)
print('y_pred = ', y_test.data)