import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.time_utils import DeformNetwork
import os
from utils.system_utils import searchForMaxIteration
from utils.general_utils import get_expon_lr_func


class DeformModel:
    def __init__(self, is_blender=False, is_6dof=False):
        self.deform = DeformNetwork(is_blender=is_blender, is_6dof=is_6dof).cuda()
        self.optimizer = None
        self.spatial_lr_scale = 5

    def step(self, xyz, time_emb):
        return self.deform(xyz, time_emb)

    '''
    这段代码定义了一个优化器和一个学习率调度器，用于在训练过程中调整模型参数的学习率。
    优化器的参数组被设置为self.deform模型的参数，学习率调度器则根据提供的参数定义了学习率随训练步数的变化。
    '''
    def train_setting(self, training_args):

        l = [
            {'params': list(self.deform.parameters()),  # 计算模型的参数，返回一个关于模型参数的迭代器。
                                                        # deform的输入参数实际为 h = torch.cat([x_emb, t_emb], dim=-1)
                                                        # 21+63 = 84 拼接为初始输入
             'lr': training_args.position_lr_init * self.spatial_lr_scale,  # 学习率
             "name": "deform"}
        ]
        '''
        self.optimizer = torch.optim.Adam(l, lr=0.0, eps=1e-15)：
        这里创建了一个Adam优化器:
        l是传递给优化器的参数组列表。
        lr=0.0设置初始学习率为0，这意味着优化器实际上不会更新参数，直到学习率调度器更新了学习率。
        eps=1e-15是一个小的数值，用于防止除以零的错误。'''
        self.optimizer = torch.optim.Adam(l, lr=0.0, eps=1e-15)

        '''
        self.deform_scheduler_args = get_expon_lr_func(...)：
        这行代码调用了一个函数get_expon_lr_func，用于创建一个指数学习率调度器。
        lr_init和lr_final分别是学习率的初始值和最终值。 
        position_lr_init = 0.00016  position_lr_final = 0.0000016
        scaling_lr = 0.001
        lr_delay_mult是一个乘数，用于延迟学习率的衰减。 position_lr_delay_mult = 0.01
        max_steps是训练的最大步数。 deform_lr_max_steps = 40_000
        返回的调度器被赋值给self.deform_scheduler_args，在训练过程中用于更新学习率。'''
        self.deform_scheduler_args = get_expon_lr_func(lr_init=training_args.position_lr_init * self.spatial_lr_scale,
                                                       lr_final=training_args.position_lr_final,
                                                       lr_delay_mult=training_args.position_lr_delay_mult,
                                                       max_steps=training_args.deform_lr_max_steps)

    def save_weights(self, model_path, iteration):
        out_weights_path = os.path.join(model_path, "deform/iteration_{}".format(iteration))
        os.makedirs(out_weights_path, exist_ok=True)
        torch.save(self.deform.state_dict(), os.path.join(out_weights_path, 'deform.pth'))

    def load_weights(self, model_path, iteration=-1):
        if iteration == -1:
            loaded_iter = searchForMaxIteration(os.path.join(model_path, "deform"))
        else:
            loaded_iter = iteration
        weights_path = os.path.join(model_path, "deform/iteration_{}/deform.pth".format(loaded_iter))
        self.deform.load_state_dict(torch.load(weights_path))

    def update_learning_rate(self, iteration):
        for param_group in self.optimizer.param_groups:
            if param_group["name"] == "deform":
                lr = self.deform_scheduler_args(iteration)
                param_group['lr'] = lr
                return lr
