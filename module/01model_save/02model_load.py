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

# 方式1
import torchvision.models
from torch import nn

model = torch.load("../../pth/vgg16_method1.pth")
print(model)

# VGG(
#   (features): Sequential(
#     (0): Conv2d(3, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
#     (1): ReLU(inplace=True)

# 方式2: 输出是个字典
model = torch.load("../../pth/vgg16_method2.pth")
print(model)
# OrderedDict([('features.0.weight', tensor([[[[ 0.0166, -0.0475,  0.0280],
#           [-0.0281, -0.0843,  0.0382],
#           [-0.0265, -0.1130, -0.0962]],

### 加载参数
vgg16 = torchvision.models.vgg16(pretrained=False)
vgg16.load_state_dict(torch.load("../../pth/vgg16_method2.pth"))
print(vgg16)

# 陷阱(解决方法：把class Mymodule 添加进来)
## 解决方法1
class Mymodule(nn.Module):
    def __init__(self):
        super(Mymodule, self).__init__()
        self.conv1 = nn.Conv2d(3,64,kernel_size=3)

    def forward(self,x):
        output = self.conv1(x)
        return output
## 解决方法2
# from 01model_save import *

model = torch.load("../../pth/mymodule_method2.pth")
print(model) # AttributeError: Can't get attribute 'Mymodule' on <module '__main__'