"""
@Description: 修改vgg16(1000分类)-->10分类
@Author   : HCQ
@Contact_1: 1756260160@qq.com
@Project  : pytorch
@File     : 00model_pretrained
@Time     : 2022/5/23 下午12:56
@Last Modify Time      @Version     @Desciption
--------------------       --------        -----------
2022/5/23 下午12:56        1.0             None
"""
import torchvision.models

# ImageNMet
# torchvision.datasets.ImageNet("../data", split=True, download=True,transform=torchvision.transforms.ToTensor())
from torch import nn
from torch.utils.data import DataLoader

vgg16_false = torchvision.models.vgg16(pretrained=False)
vgg16_true = torchvision.models.vgg16(pretrained=True) # 已经训练好的权重

vgg16_true.classifier.add_module("add_linear", nn.Linear(1000, 10)) # 添加新的层
vgg16_false.classifier[6] = nn.Linear(in_features=4096, out_features=10)  # 修改新的层

print(vgg16_false)
#
train_data = torchvision.datasets.CIFAR10("../data", train=True, transform=torchvision.transforms.ToTensor(),
                                          download=True)
dataloader = DataLoader(train_data, batch_size = 64)


#
# for data in dataloader:
#     imgs, targets = data
#     output = vgg16_true(imgs)
