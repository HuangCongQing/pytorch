'''
Description: 多分类问题(softmax)
视频：https://www.bilibili.com/video/BV1Y7411d7Ys?p=9
博客：https://blog.csdn.net/bit452/article/details/109686936
Author: HCQ
Company(School): UCAS
Email: 1756260160@qq.com
Date: 2020-12-06 20:05:38
LastEditTime: 2020-12-06 21:34:51
FilePath: /pytorch/PyTorch深度学习实践/09多分类问题softmax.py
'''
import torch
from torchvision import transforms # 针对图像处理
from torchvision import datasets
from torch.utils.data import DataLoader
import torch.nn.functional as F # 为了使用激活函数relu()
import torch.optim as optim # optim.SGD
 
# 1 prepare dataset
 
batch_size = 64
#  归一化,均值和标准差  将PIL Image转化为Tensor  # 0.1307,), (0.3081,) 分别对应均值和标准差
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]) 
 
train_dataset = datasets.MNIST(root='./data/mnist/', train=True, download=True, transform=transform)
train_loader = DataLoader(train_dataset, shuffle=True, batch_size=batch_size)
test_dataset = datasets.MNIST(root='./data/mnist/', train=False, download=True, transform=transform)
test_loader = DataLoader(test_dataset, shuffle=False, batch_size=batch_size)
 
# 2 design model using class
 
 
class Net(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.l1 = torch.nn.Linear(784, 512)
        self.l2 = torch.nn.Linear(512, 256)
        self.l3 = torch.nn.Linear(256, 128)
        self.l4 = torch.nn.Linear(128, 64)
        self.l5 = torch.nn.Linear(64, 10)
 
    def forward(self, x):
        x = x.view(-1, 784)  # -1其实就是自动获取mini_batch，即N，mini样本数
        x = F.relu(self.l1(x)) # 激活函数relu
        x = F.relu(self.l2(x))
        x = F.relu(self.l3(x))
        x = F.relu(self.l4(x))
        return self.l5(x)  # 最后一层不做激活，不进行非线性变换
 
 
model = Net()
 
# 3 construct loss and optimizer
criterion = torch.nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.5)
 
# 4 training cycle forward, backward, update
 
 
def train(epoch):
    running_loss = 0.0
    for batch_idx, data in enumerate(train_loader, 0):
        inputs, target = data
        optimizer.zero_grad() # 优化器清零
 
        outputs = model(inputs) # 计算y hat
        loss = criterion(outputs, target) # 计算loss
        loss.backward()
        optimizer.step()
 
        running_loss += loss.item() # items()
        if batch_idx % 300 == 299: # 训练300次输出一次
            print('[%d, %5d] loss: %.3f' % (epoch+1, batch_idx+1, running_loss/300))
            running_loss = 0.0
 
 # 测试（不需要反向传播，主需要算正向的）
def test():
    correct = 0
    total = 0
    with torch.no_grad(): # 来阻止autograd跟踪设置了 .requires_grad=True 的张量的历史记录。
        for data in test_loader: # test_loader 测试集
            images, labels = data
            outputs = model(images)  # torch.Size([64, 10])
            _, predicted = torch.max(outputs.data, dim=1) # dim = 1 列是第0个维度，行是第1个维度  求每一行最大下标  （64，1） ||    _,对应最大值，predicted对应最大值索引
            total += labels.size(0) #  labels.size(0) = N  ,   (N, 1)  
            correct += (predicted == labels).sum().item() # 张量之间的比较运算  预测值和label值
    print('accuracy on test set: %d %% ' % (100*correct/total))
 
 
if __name__ == '__main__':
    for epoch in range(10):
        train(epoch)
        #  if epoch % 10 == 9: # 可以设置10轮测试一次
        test()