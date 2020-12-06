'''
Description: 处理多维特征的输入
视频：https://www.bilibili.com/video/BV1Y7411d7Ys?p=7
博客：
https://blog.csdn.net/bit452/article/details/109682078
https://blog.csdn.net/weixin_44841652/article/details/105125826
Author: HCQ
Company(School): UCAS
Email: 1756260160@qq.com
Date: 2020-12-06 12:37:48
LastEditTime: 2020-12-06 17:16:16
FilePath: /pytorch/PyTorch深度学习实践/07处理多维特征的输入.py
'''

 
import  torch
# import  torch.nn.functional as F  # 没用到
import  numpy as np
import matplotlib.pyplot as plt  # 画图
# from sklearn import datasets  # 没用到

#  prepare dataset
xy=np.loadtxt('./data/Diabetes_class.csv.gz',delimiter=',',dtype=np.float32)#加载训练集合
x_data = torch.from_numpy(xy[:,:-1])#取前八列  第二个‘:-1’是指从第一列开始，最后一列不要
y_data = torch.from_numpy(xy[:,[-1]]) # [-1] 最后得到的是个矩阵

# 没有这个测试集
# test =np.loadtxt('./data/test_class.csv.gz',delimiter=',',dtype=np.float32)#加载测试集合，这里我用数据集的最后一个样本做测试，训练集中没有最后一个样本
# test_x = torch.from_numpy(test)

# 2 design model using class
class Model(torch.nn.Module):
    def __init__(self):#构造函数
        super(Model,self).__init__()
        self.linear1 = torch.nn.Linear(8,6)#8维到6维
        self.linear2 = torch.nn.Linear(6, 4)#6维到4维
        self.linear3 = torch.nn.Linear(4, 1)#4维到1维
        self.sigmoid = torch.nn.Sigmoid()# 将其看作是网络的一层，而不是简单的函数使用 # 因为他里边也没有权重需要更新，所以要一个就行了，单纯的算个数
        # 尝试不同的激活函数  torch.nn.ReLU()


    def forward(self, x):#构建一个计算图，就像上面图片画的那样
        x = self.sigmoid(self.linear1(x))
        x = self.sigmoid(self.linear2(x))#将上面一行的输出作为输入
        x = self.sigmoid(self.linear3(x)) # # y hat  ==================================
        return  x


# 3  construct loss and optimizer
model = Model()#实例化模型
criterion = torch.nn.BCELoss(size_average=False)
#model.parameters()会扫描module中的所有成员，如果成员中有相应权重，那么都会将结果加到要训练的参数集合上
optimizer = torch.optim.SGD(model.parameters(),lr=0.1)#lr为学习率，因为0.01太小了，我改成了0.1


epoch_list = [] # 用来画图
loss_list = [] # 用来画图
# 4 training cycle forward, backward, update
for epoch in range(1000):
    #Forward
    y_pred = model(x_data)   # 没有用Mini_batch
    loss = criterion(y_pred,y_data)
    print(epoch,loss.item())
    epoch_list.append(epoch) # 用来画图
    loss_list.append(loss.item()) # 用来画图
    #Backward
    optimizer.zero_grad()
    loss.backward()
    #update
    optimizer.step()

y_pred = model(x_data)

print(y_pred.detach().numpy())

# y_pred2 = model(test_x)
# print(y_pred2.data.item())

# 绘图
plt.plot(epoch_list, loss_list)
plt.ylabel('loss')
plt.xlabel('epoch')
plt.show()

