'''
Description: 加载数据集
Author: HCQ
Company(School): UCAS
Email: 1756260160@qq.com
Date: 2020-12-06 17:24:26
LastEditTime: 2020-12-06 17:29:45
FilePath: /pytorch/PyTorch深度学习实践/08加载数据集.py
'''

import  torch
import  numpy as np
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

'''
Dataset是一个抽象函数，不能直接实例化，所以我们要创建一个自己类，继承Dataset
继承Dataset后我们必须实现三个函数：
__init__()是初始化函数，之后我们可以提供数据集路径进行数据的加载
__getitem__()帮助我们通过索引找到某个样本
__len__()帮助我们返回数据集大小
'''
class DiabetesDataset(Dataset):
    def __init__(self,filepath):
        xy = np.loadtxt(filepath,delimiter=',',dtype=np.float32)
        #shape本身是一个二元组（x,y）对应数据集的行数和列数，这里[0]我们取行数,即样本数
        self.len = xy.shape[0]
        self.x_data = torch.from_numpy(xy[:, :-1])
        self.y_data = torch.from_numpy(xy[:, [-1]])

    def __getitem__(self, index):
        return self.x_data[index],self.y_data[index]

    def __len__(self):
        return self.len
        
#定义好DiabetesDataset后我们就可以实例化他了
dataset = DiabetesDataset('./data/Diabetes_class.csv.gz')
#我们用DataLoader为数据进行分组，batch_size是一个组中有多少个样本，shuffle表示要不要对样本进行随机排列
#一般来说，训练集我们随机排列，测试集不。num_workers表示我们可以用多少进程并行的运算
train_loader = DataLoader(dataset=dataset,batch_size=32,shuffle=True,num_workers=2)

class Model(torch.nn.Module):
    def __init__(self):#构造函数
        super(Model,self).__init__()
        self.linear1 = torch.nn.Linear(8,6)#8维到6维
        self.linear2 = torch.nn.Linear(6, 4)#6维到4维
        self.linear3 = torch.nn.Linear(4, 1)#4维到1维
        self.sigmoid = torch.nn.Sigmoid()#因为他里边也没有权重需要更新，所以要一个就行了，单纯的算个数


    def forward(self, x):#构建一个计算图，就像上面图片画的那样
        x = self.sigmoid(self.linear1(x))
        x = self.sigmoid(self.linear2(x))
        x = self.sigmoid(self.linear3(x))
        return  x

model = Model()#实例化模型

criterion = torch.nn.BCELoss(size_average=False)
#model.parameters()会扫描module中的所有成员，如果成员中有相应权重，那么都会将结果加到要训练的参数集合上
optimizer = torch.optim.SGD(model.parameters(),lr=0.1)#lr为学习率

if __name__=='__main__':#if这条语句在windows系统下一定要加，否则会报错
    for epoch in range(1000):
        # 循环所有的epoch
        for i,data in enumerate(train_loader,0):#取出一个bath # 嵌套循环，没执行一次迭代，执行1个mini-batch
            # repare data
            inputs,labels = data#将输入的数据赋给inputs，结果赋给labels 
            #Forward
            y_pred = model(inputs)
            loss = criterion(y_pred,labels)
            print(epoch,loss.item())
            #Backward
            optimizer.zero_grad()
            loss.backward()
            #update
            optimizer.step()




