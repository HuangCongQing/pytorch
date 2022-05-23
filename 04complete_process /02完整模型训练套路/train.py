"""
@Description: 训练完整流程
@Author   : HCQ
@Contact_1: 1756260160@qq.com
@Project  : pytorch
@File     : train.py
@Time     : 2022/5/23 下午4:19
@Last Modify Time      @Version     @Desciption
--------------------       --------        -----------
2022/5/23 下午4:19        1.0             None
"""
from model import *
import torchvision.datasets
from torch import nn
from torch.utils.data import DataLoader


# 1 准备数据集
train_data = torchvision.datasets.CIFAR10("../../data", train=True, transform=torchvision.transforms.ToTensor(),
                                          download=True)
test_data = torchvision.datasets.CIFAR10("../../data", train=False, transform=torchvision.transforms.ToTensor(),
                                          download=True)
print("训练集数据长度：%d \n测试集数据长度：%d"%(len(train_data), len(test_data)))
# 训练集数据长度：50000
# 测试集数据长度：10000
train_dataloader = DataLoader(train_data, batch_size=64)
test_dataloader = DataLoader(test_data, batch_size=64)


# 2 model
mymodule = Mymodule()

# 3 损失函数和优化器
loss_fn = nn.CrossEntropyLoss()
optimizar = torch.optim.SGD(mymodule.parameters(), lr=1e-2)

# 4 训练 circle
## 记录训练的次数
total_train_step = 0
## 记录测试的次数
total_test_step = 0
## 训练论数
epoch = 10

for i in range(epoch):
    print("------训练第%d轮"%(i+1))

    for data in train_dataloader:
        imgs, target = data
        output = mymodule(imgs)
        loss = loss_fn(output, target)

        # 优化器
        optimizar.zero_grad()
        loss.backward()
        optimizar.step()

        total_train_step +=1
        print("训练次数:%d, Loss:%f"%(total_train_step, loss.item()))

