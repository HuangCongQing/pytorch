"""
@Description: 优化器 不是nn里的模块，而是torch.torch.optim.SGD(mymodule.parameters(), lr=0.01) 
@Author   : HCQ
@Contact_1: 1756260160@qq.com
@Project  : pytorch
@File     : 08torch.optim
@Time     : 2022/5/22 下午10:12
@Last Modify Time      @Version     @Desciption
--------------------       --------        -----------
2022/5/22 下午10:12        1.0             None
"""

import torch
import torchvision
from torch import nn
from torch.nn import Conv2d, MaxPool2d, Flatten, Linear, Sequential
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter


dataset = torchvision.datasets.CIFAR10("../data", train=False, download=True, transform=torchvision.transforms.ToTensor())
dataloader = DataLoader(dataset,batch_size=64)


class Mymodule(nn.Module):
    def __init__(self):
        super(Mymodule, self).__init__()
        # Sequential
        self.model1 = Sequential(
            Conv2d(3, 32, 5, padding=2),
            MaxPool2d(2),
            Conv2d(32, 32, 5, padding=2),
            MaxPool2d(2),
            Conv2d(32, 64, 5, padding=2),
            MaxPool2d(2),
            Flatten(),
            Linear(1024, 64),
            Linear(64, 10)  # 10分类
        )

    def forward(self,x):
        x = self.model1(x)
        return x

mymodule = Mymodule()
# print(mymodule) # 输出网络结构
loss = nn.CrossEntropyLoss()
optim = torch.optim.SGD(mymodule.parameters(), lr=0.01) # 优化器=================

for epoch in range(20):
    running_loss = 0.
    for data in dataloader:
        imgs, targets = data
        output = mymodule(imgs)
        result_loss = loss(output, targets)
        #
        optim.zero_grad()
        result_loss.backward() # 反向传播得到梯度~~~~~~~~~~~~~~
        optim.step() # 更新权重weight.data 对每个参数进行调优

        # print(result_loss)
        running_loss = running_loss + result_loss
    print(running_loss)




# input = torch.ones(64,3,32,32) # 模拟数据测试网络结构
# output = mymodule(input)
# print(output.shape) # torch.Size([64, 10])
