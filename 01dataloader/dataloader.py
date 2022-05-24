"""
@Description: dataloader使用 https://www.bilibili.com/video/BV1hE411t7RN?p=15&t=343.6
@Author   : HCQ
@Contact_1: 1756260160@qq.com
@Project  : pytorch
@File     : dataloader
@Time     : 2022/5/24 下午4:20
@Last Modify Time      @Version     @Desciption
--------------------       --------        -----------
2022/5/24 下午4:20        1.0             None
"""
import torchvision.datasets
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

dataset = torchvision.datasets.CIFAR10("../data", train=True, transform=torchvision.transforms.ToTensor(),
                                       download=True)
#
dataloader = DataLoader(dataset, batch_size=64, shuffle=True, num_workers=0, drop_last=True) # drop_last=True 表示最后一个bath舍弃

writer = SummaryWriter("../logs/dataloader")
for epoch in range(2):
    step=0
    for data in dataloader:
        imgs, target = data
        print(imgs.shape)
        print(target)
        writer.add_images("Epoch_%d"%(epoch), imgs, step)
        step +=1
writer.close()