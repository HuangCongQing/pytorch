"""
@Description: GPU常用方式
@Author   : HCQ
@Contact_1: 1756260160@qq.com
@Project  : pytorch
@File     : train_gpu_2
@Time     : 2022/5/23 下午8:47
@Last Modify Time      @Version     @Desciption
--------------------       --------        -----------
2022/5/23 下午8:47        1.0             None
"""




import torch
from torch.utils.tensorboard import SummaryWriter

# from model import *
import torchvision.datasets
from torch import nn
from torch.utils.data import DataLoader

# 定义训练的设备!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
# device = torch.device("cpu")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("设备：", device)

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
mymodule = Mymodule()
# ========================cuda======================================
# if torch.cuda.is_available():
#     mymodule = mymodule.cuda()
mymodule = mymodule.to(device) # 也可以不用赋值 直接mymodule.to(device)

# 3 损失函数和优化器
loss_fn = nn.CrossEntropyLoss()
# ========================cuda======================================
# if torch.cuda.is_available():
#     loss_fn = loss_fn.cuda()
loss_fn = loss_fn.to(device) # 也可以不用赋值 直接loss_fn.to(device)

optimizar = torch.optim.SGD(mymodule.parameters(), lr=1e-2)

# 4 训练 circle
## 记录训练的次数
total_train_step = 0
## 记录测试的次数
total_test_step = 0
## 训练论数
epoch = 10

# 添加tensorboard
writer = SummaryWriter("../../logs/loss")

for i in range(epoch):
    print("------训练第%d轮--------"%(i+1))

    # 训练步骤开始
    mymodule.train() # 非必要
    for data in train_dataloader:
        imgs, target = data
        # # ========================cuda======================================
        # if torch.cuda.is_available():
        #     imgs = imgs.cuda()
        #     target = target.cuda()
        imgs = imgs.to(device)  # 必须这种
        target = target.to(device) # 必须这种

        output = mymodule(imgs)
        loss = loss_fn(output, target)

        # 优化器
        optimizar.zero_grad()
        loss.backward()
        optimizar.step()

        total_train_step +=1
        if total_train_step % 100 == 0:
            print("训练次数:%d, Loss:%f"%(total_train_step, loss.item()))
            writer.add_scalar("train_loss", loss.item(), total_train_step)


    # 测试步骤开始
    # 用with torch.no_grad():表明当前计算不需要反向传播，使用之后，强制后边的内容不进行计算图的构建
    # 用于禁用梯度计算功能，以加快计算速度
    mymodule.eval() # 非必要
    total_test_loss = 0
    total_test_accuracy = 0
    with torch.no_grad():
        for data in test_dataloader:
            imgs, target = data
            # ========================cuda======================================
            if torch.cuda.is_available():
                imgs = imgs.cuda()
                target = target.cuda()
            output = mymodule(imgs)
            loss = loss_fn(output, target)
            total_test_loss +=loss.item()
            accuracy = (output.argmax()==target).sum()
            total_test_accuracy +=accuracy.item()
        print("整体测试集loss: %f"%(total_test_loss))
        print("整体测试集正确率：%f"%(total_test_accuracy/len(test_data)))
        writer.add_scalar("test_loss", total_test_loss, total_test_step)
        writer.add_scalar("test_accuracy", (total_test_accuracy)/len(test_data), total_test_step)

        total_test_step +=1

    # 保存模型
    torch.save(mymodule, "../../pth/module_all/module_%d.pth"%(i))
    # torch.save(mymodule.state_dict(), "../../pth/module_all/module_%d.pth"%(i))
    print("模型已保存")

writer.close()
