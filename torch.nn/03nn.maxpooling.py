"""
@Description:
@Author   : HCQ
@Contact_1: 1756260160@qq.com
@Project  : pytorch
@File     : 03nn.maxpooling.py
@Time     : 2022/5/22 下午5:09
@Last Modify Time      @Version     @Desciption
--------------------       --------        -----------
2022/5/22 下午5:09        1.0             None
"""

import torch

# torch.nn.MaxPool2d(kernel_size, stride=None, padding=0, dilation=1, return_indices=False, ceil_mode=False)
import torchvision.datasets
from torch import nn
from torch.nn import MaxPool2d

# input = torch.tensor([[1,2,3,4,5],
#                         [1,2,3,4,5],
#                         [1,2,3,4,5],
#                         [1,2,3,4,5],
#                         [1,2,3,4,5],], dtype=torch.float32)
# # 转成BCWH
# input = torch.reshape(input, (-1, 1,5,5))
# print(input.shape)
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

dataset = torchvision.datasets.CIFAR10("../data", train=False, download=True, transform=torchvision.transforms.ToTensor())
dataloader = DataLoader(dataset,batch_size=64)


class Mymodule(nn.Module):
    def __init__(self):
        super(Mymodule, self).__init__()
        self.maxpool1 = MaxPool2d(kernel_size=3, ceil_mode=False)
        # self.maxpool1 = MaxPool2d(kernel_size=3, ceil_mode=True) # 向上取整
        # torch.nn.MaxPool2d(kernel_size, stride=None, padding=0, dilation=1, return_indices=False, ceil_mode=False)
    def forward(self, x):
        output = self.maxpool1(x)
        return output

mymodule = Mymodule()

# output = mymodule(input)
# print(output.shape)

writer = SummaryWriter("../logs/maxpool")
step = 0
for data in dataloader:
    imgs, targets = data
    writer.add_images("input", imgs, step)
    output = mymodule(imgs)
    writer.add_images("output", output, step)
    step +=1

writer.close()
