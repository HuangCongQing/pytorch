"""
@Description: 
@Author   : HCQ
@Contact_1: 1756260160@qq.com
@Project  : pytorch
@File     : model
@Time     : 2022/5/23 下午4:40
@Last Modify Time      @Version     @Desciption
--------------------       --------        -----------
2022/5/23 下午4:40        1.0             None
"""
import torch
from torch import nn


# 2 model
class Mymodule(nn.Module):
    def __init__(self):
        super(Mymodule, self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(3,32,5,1,2),
            nn.MaxPool2d(2),
            nn.Conv2d(32,32,5,1,2),
            nn.MaxPool2d(2),
            nn.Conv2d(32,64,5,1,2),
            nn.MaxPool2d(2),
            nn.Flatten(), #展躺
            nn.Linear(64*4*4, 64),
            nn.Linear(64, 10),

        )

    def forward(self, x):
        output = self.model(x)
        return output

if __name__ == '__main__':
    mymodule = Mymodule()
    input = torch.ones((64,3,32,32))
    output = mymodule(input)
    print(output.shape) # torch.Size([64, 10])