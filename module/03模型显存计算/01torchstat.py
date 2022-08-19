'''
Description: https://blog.csdn.net/jining11/article/details/89947541?
Author: HCQ
Company(School): UCAS
Email: 1756260160@qq.com
Date: 2022-08-12 11:07:04
LastEditTime: 2022-08-12 11:10:26
FilePath: /pytorch/module/03模型显存计算/01torchstat.py
'''
from torchstat import stat
import torchvision.models as models
model = models.resnet152()
stat(model, (3, 224, 224))
''' 
Total params: 60,192,808
---------------------------------------------------------------------------------------------------------------------------------------------------------------------
Total memory: 226.06MB
Total MAdd: 23.11GMAdd
Total Flops: 11.57GFlops
Total MemR+W: 682.3MB

 '''