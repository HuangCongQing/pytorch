"""
@Description: 
@Author   : HCQ
@Contact_1: 1756260160@qq.com
@Project  : pytorch
@File     : 01model_load
@Time     : 2022/5/23 下午1:29
@Last Modify Time      @Version     @Desciption
--------------------       --------        -----------
2022/5/23 下午1:29        1.0             None
"""

import torch
import torchvision
from torch import nn

vgg16 = torchvision.models.vgg16(pretrained=False)

# 保存方式1: 模型结构+模型参数
torch.save(vgg16, "../../pth/vgg16_method1.pth")

# 保存方式2：（官方推荐）只保存参数，格式字典dict形式
torch.save(vgg16.state_dict(), "../../pth/vgg16_method2.pth")


# 陷阱
class Mymodule(nn.Module):
    def __init__(self):
        super(Mymodule, self).__init__()
        self.conv1 = nn.Conv2d(3,64,kernel_size=3)

    def forward(self,x):
        output = self.conv1(x)
        return output

mymodule = Mymodule()
torch.save(mymodule, "../../pth/mymodule_method2.pth")
print("导出完成")