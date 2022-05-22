"""
@Description: 
@Author   : HCQ
@Contact_1: 1756260160@qq.com
@Project  : pytorch
@File     : 07nn.L1Loss
@Time     : 2022/5/22 下午9:34
@Last Modify Time      @Version     @Desciption
--------------------       --------        -----------
2022/5/22 下午9:34        1.0             None
"""

import torch
from torch import nn
from torch.nn import L1Loss

input = torch.tensor([1,2,3], dtype=torch.float32)
target = torch.tensor([1,2,5], dtype=torch.float32)

input = torch.reshape(input, (1,1,1,3))
target = torch.reshape(target, (1,1,1,3))

loss = L1Loss(reduction="mean")
result = loss(input, target)

loss_mse = nn.MSELoss()
result_mse = loss_mse(input, target)
print(result)
print(result_mse)

# 交叉熵-----

x = torch.tensor([0.1,0.2,0.3])
y = torch.tensor([1])
x = torch.reshape(x, (1,3)) # 输入有要求
loss_cross = nn.CrossEntropyLoss()
result = loss_cross(x, y)
print(result)