'''
Description: pth权重分析  https://www.yuque.com/huangzhongqing/pytorch/ayz76a
Author: HCQ
Company(School): UCAS
Email: 1756260160@qq.com
Date: 2022-10-25 19:02:45
LastEditTime: 2022-10-25 19:03:16
FilePath: /pytorch/module/04pth_onnx_analysis/01pth_analysis.py
'''

import torch
import torch.distributed as dist

# https://www.cnblogs.com/tnak-liu/p/14615047.html
"""
        pytorch 分布式训练初始化
        1) backend (str): 指定通信所用后端，可以是'ncll'、'gloo' 或者是一个torch.ditributed.Backend类
        2) init_method (str): 这个URL指定了如何初始化互相通信的进程
        3) world_size (int): 执行训练的所有的进程数   等于 cuda.device 数量
        4) rank (int): 进程的编号，即优先级
"""

# local_rank = 0
# torch.distributed.init_process_group(backend="nccl", init_method='tcp://10.1.1.20:23456', world_size=torch.cuda.device_count(),
#                                      rank=local_rank)

is_cuda= True
pthfile = '/home/chongqinghuang/data/weights/rpv_with_padding_60epochs_bs2_miou67.31.pt'
init_net = torch.load(pthfile,
                  # map_location='cuda:%d' % dist.local_rank()
                  map_location='cuda:0'
                  if is_cuda else 'cpu')  # 修改

print("type: ", type(init_net))  # 类型是 dict
print("len(init_net): ", len(init_net))  # 长度为 4，即存在四个 key-value 键值对

for k, v in init_net['model'].items():
    if k.startswith('module.voxel_down1.layer0.conv'):
        print(k, v.shape)  # 查看四个键，分别是 model,optimizer,scheduler,iteration
        # print(v)
# module.voxel_down1.layer0.conv.kernel torch.Size([8, 32, 32])
# module.voxel_down1.layer0.conv.bn_weight torch.Size([32])
# module.voxel_down1.layer0.conv.bn_bias torch.Size([32])
# module.voxel_down1.layer0.conv.running_mean torch.Size([32])
# module.voxel_down1.layer0.conv.running_var torch.Size([32])
# module.voxel_down1.layer0.conv.num_batches_tracked torch.Size([])

print("end!")
