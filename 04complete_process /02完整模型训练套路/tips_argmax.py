"""
@Description: 
@Author   : HCQ
@Contact_1: 1756260160@qq.com
@Project  : pytorch
@File     : tips_argmax
@Time     : 2022/5/23 下午7:25
@Last Modify Time      @Version     @Desciption
--------------------       --------        -----------
2022/5/23 下午7:25        1.0             None
"""

import torch

output = torch.tensor([[0.2,  0.5],
                       [0.4, 0.2]])
print(output.argmax(1)) # tensor([1, 0])
pred = output.argmax(1)
target = torch.tensor([1,0])
print((pred == target).sum()) # tensor([True, True])