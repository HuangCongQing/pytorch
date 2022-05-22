"""
@Description: 25-->3 torch.nn.Linear(in_features, out_features, bias=True, device=None, dtype=None)
@Author   : HCQ
@Contact_1: 1756260160@qq.com
@Project  : pytorch
@File     : 05nn.Linear.py
@Time     : 2022/5/22 下午7:40
@Last Modify Time      @Version     @Desciption
--------------------       --------        -----------
2022/5/22 下午7:40        1.0             None
"""
import torch

# torch.nn.Linear(in_features, out_features, bias=True, device=None, dtype=None)
import torchvision.datasets
from torch import nn
from torch.nn import Linear
from torch.utils.data import DataLoader

dataset = torchvision.datasets.CIFAR10("../data", train=False, transform=torchvision.transforms.ToTensor())
dataloader = DataLoader(dataset, batch_size=64)

class Mymodule(nn.Module):
    def __init__(self):
        super(Mymodule, self).__init__()
        self.fc1 = Linear(196608, 10) # torch.nn.Linear(in_features, out_features, bias=True, device=None, dtype=None)
    def forward(self, x):
        output = self.fc1(x)
        return output

mymodule = Mymodule()


for data in dataloader:
    imgs, targets = data
    print(imgs.shape) # torch.Size([16, 3, 32, 32])
    # output = torch.reshape(imgs, (1,1,1,-1)) torch.Size([1, 1, 1, 49152])
    output = torch.flatten(imgs) # 展平 torch.Size([196608])
    print(output.shape) #  torch.Size([196608])
    output = mymodule(output)
    print(output.shape) # torch.Size([1, 1, 1, 10])

# torch.Size([64, 3, 32, 32])
# torch.Size([1, 1, 1, 196608])
# torch.Size([1, 1, 1, 10])