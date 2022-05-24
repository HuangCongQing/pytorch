"""
@Description: 制作数据集 https://www.bilibili.com/video/BV1hE411t7RN?p=7&spm_id_from=pageDriver
@Author   : HCQ
@Contact_1: 1756260160@qq.com
@Project  : pytorch
@File     : read_data
@Time     : 2022/5/24 上午10:55
@Last Modify Time      @Version     @Desciption
--------------------       --------        -----------
2022/5/24 上午10:55        1.0             None
"""
from torch.utils.data import DataLoader
import os
from PIL import Image

'''
Dataset是一个抽象函数，不能直接实例化，所以我们要创建一个自己类，继承Dataset
继承Dataset后我们必须实现三个函数：
__init__()是初始化函数，之后我们可以提供数据集路径进行数据的加载
__getitem__()帮助我们通过索引找到某个样本
__len__()帮助我们返回数据集大小
'''
class MyData(DataLoader):
    def __init__(self, root_dir, label_dir):
        # 指定类当中的全局变量
        self.root_dir = root_dir
        self.label_dir = label_dir
        self.path = os.path.join(self.root_dir, self.label_dir)
        self.img_path = os.listdir(self.path)


    def __getitem__(self, idx):
        img_name = self.img_path[idx] # '466430434_4000737de9.jpg',
        img_item_name = os.path.join(self.root_dir, self.label_dir, img_name) # 完整路径
        # 得到图像
        image = Image.open(img_item_name)
        # 得到label
        label = self.label_dir # 这个文件夹的名字就算label

        return image, label

    def __len__(self):
        return len(self.img_path)




#
root_dir = "../imgs/hymenoptera_data/train"
ant_label_dir = "ants"
bees_label_dir = "bees"

ants_mydata = MyData(root_dir, ant_label_dir)
bees_mydata = MyData(root_dir, bees_label_dir)
print(ants_mydata[0])
# imgs, label = ants_mydata[0]
# imgs.show()
print(len(ants_mydata), len(bees_mydata))
# train_mydata = ants_mydata + bees_mydata # TypeError: unsupported operand type(s) for +: 'MyData' and 'MyData'

# for data in ants_mydata:
#     imgs, target = data
#     print(target)

for i in range(len(ants_mydata)):
    imgs, target = ants_mydata[i]
    print(target)
