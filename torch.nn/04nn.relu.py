"""
@Description: 
@Author   : HCQ
@Contact_1: 1756260160@qq.com
@Project  : pytorch
@File     : 04nn.relu.py
@Time     : 2022/5/22 下午6:03
@Last Modify Time      @Version     @Desciption
--------------------       --------        -----------
2022/5/22 下午6:03        1.0             None
"""
import torch
from torch import nn
from torch.nn import ReLU

input = torch.tensor([[1,-1],
                      [4,-4]])
input = torch.reshape(input, (-1,1,2,2))
print(input.shape)

class Mymodule(nn.Module):
    def __init__(self):
        super(Mymodule, self).__init__()
        self.relu1 = ReLU() # 无参数 torch.nn.ReLU(inplace=False)

    def forward(self, x):
        output = self.relu1(x)
        return output

mymodule = Mymodule()

output = mymodule(input)
print(output)

# torch.Size([1, 1, 2, 2])
# tensor([[[[1, 0],
#           [4, 0]]]])