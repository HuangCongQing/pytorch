'''
Description: 
Author: HCQ
Company(School): UCAS
Email: 1756260160@qq.com
Date: 2022-05-22 12:15:00
LastEditTime: 2022-05-22 16:05:25
FilePath: /pytorch/torch.nn/02nn.conv2d.py
'''
import torch
import torchvision
from torch import nn
from torch.nn import Conv2d
from torch.utils.data import DataLoader

# 1 加载数据
from torch.utils.tensorboard import SummaryWriter

dataset = torchvision.datasets.CIFAR10("../data",train=False,transform=torchvision.transforms.ToTensor(),
                                       download=True)
dataloader = DataLoader(dataset, batch_size=64) # batch_size 自己设置

# 2 模型
class Mymodule(nn.Module):
    def __init__(self):
        super(Mymodule, self).__init__()
        self.conv1 = Conv2d(in_channels=3, out_channels=6, kernel_size=3, stride=1, padding=0)

    def forward(self, x):
        x = self.conv1(x)
        return x

# 调用
mymodule = Mymodule()
# print(mymodule) # 打印输出模型结构 (conv1): Conv2d(3, 6, kernel_size=(3, 3), stride=(1, 1))

writer = SummaryWriter("../logs")
step=0
for data in dataloader:
    img, target = data
    output = mymodule(img)
    print(img.shape) # torch.Size([64, 3, 32, 32])
    print(output.shape) # torch.Size([64, 6, 30, 30])
    # 添加tensorboard
    output = torch.reshape(output, (-1, 3, 30, 30))# assert I.ndim == 4 and I.shape[1] == 3
    writer.add_images("input", img, step)
    writer.add_images("output", output, step)
    step+=1