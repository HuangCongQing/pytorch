"""
@Description: transform Image格式转为Tensor   https://www.bilibili.com/video/BV1hE411t7RN?p=14
@Author   : HCQ
@Contact_1: 1756260160@qq.com
@Project  : pytorch
@File     : dataset_transform
@Time     : 2022/5/24 下午3:42
@Last Modify Time      @Version     @Desciption
--------------------       --------        -----------
2022/5/24 下午3:42        1.0             None
"""

import  torchvision

# 转格式
from torch.utils.tensorboard import SummaryWriter

data_transform = torchvision.transforms.Compose([
    torchvision.transforms.ToTensor()
])
train_set = torchvision.datasets.CIFAR10(root="../data", train=True, transform=data_transform, download=True)

print(train_set[0])

# image, target = train_set[0]
# print(train_set[0])
# print(train_set.classes[target]) # ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
# image.show()


writer = SummaryWriter("../logs/p10")
for i in range(10):
    img, target = train_set[i]
    writer.add_image("train_set", img, i)

writer.close()
