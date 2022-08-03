"""
@Description: CIFAR10分类  https://www.yuque.com/huangzhongqing/pytorch/hwk1g7#ObngJ
@Author   : HCQ
@Contact_1: 1756260160@qq.com
@Project  : pytorch
@File     : 06nn.Sequential_CIFATR10_cls
@Time     : 2022/5/22 下午8:18
@Last Modify Time      @Version     @Desciption
--------------------       --------        -----------
2022/5/22 下午8:18        1.0             None
"""
# 结构：https://www.yuque.com/huangzhongqing/pytorch/hwk1g7#ObngJ
# Conv2d(in_channels, out_channels, kernel_size, stride=1, padding=0
import torch
from torch import nn
from torch.nn import Conv2d, MaxPool2d, Flatten, Linear, Sequential
from torch.utils.tensorboard import SummaryWriter

# 流程示意图：https://www.yuque.com/huangzhongqing/pytorch/hwk1g7#ObngJ
class Mymodule(nn.Module):
    def __init__(self):
        super(Mymodule, self).__init__()
        # self.conv1 = Conv2d(3, 32, 5, padding=2)
        # self.maxpool1 = MaxPool2d(2)
        # self.conv2 = Conv2d(32, 32, 5, padding=2)
        # self.maxpool2 = MaxPool2d(2)
        # self.conv3 = Conv2d(32, 64, 5, padding=2)
        # self.maxpool3 = MaxPool2d(2)
        # self.flatten = Flatten()
        # self.linear1 = Linear(1024, 64)
        # self.linear2 = Linear(64, 10) # 10分类
        
        # Sequential
        self.model1 = Sequential(
            # input:(64,3,32,32) BCHW  计算公式（n+2p-f）+1
            Conv2d(3, 32, 5, padding=2), # (64,32,32,32) Conv2d(in_channels, out_channels, kernel_size, stride=1, padding=0
            MaxPool2d(2),                # (64,32,16,16)   
            Conv2d(32, 32, 5, padding=2), # (64,32,16,16)
            MaxPool2d(2),                # (64,32,8,8)
            Conv2d(32, 64, 5, padding=2), # (64,64,8,8)
            MaxPool2d(2),                # (64,64,4,4) 64X4X4 = 1024
            Flatten(),         # （64，1024）
            Linear(1024, 64),  # （64，64）
            Linear(64, 10)  # 10分类 # （64，10）
        )

    def forward(self,x):
        # x = self.conv1(x)
        # x = self.maxpool1(x)
        # x = self.conv2(x)
        # x = self.maxpool2(x)
        # x = self.conv3(x)
        # x = self.maxpool3(x)
        # x = self.flatten(x)
        # x = self.linear1(x)
        # x = self.linear2(x)
        x = self.model1(x)
        return x

mymodule = Mymodule()
print(mymodule) # 输出网络结构

input = torch.ones(64,3,32,32) # 模拟数据测试网络结构(BCHW)
output = mymodule(input)
print(output.shape) # torch.Size([64, 10])

# 查看网络结构
writer = SummaryWriter("../logs/graph")
writer.add_graph(mymodule, input)
writer.close()
