import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.rigid_utils import exp_se3


def get_embedder(multires, i=1):
    if i == -1:
        return nn.Identity(), 3

    embed_kwargs = {
        'include_input': True,
        'input_dims': i,
        'max_freq_log2': multires - 1,
        'num_freqs': multires,
        'log_sampling': True,
        'periodic_fns': [torch.sin, torch.cos],
    }

    embedder_obj = Embedder(**embed_kwargs)
    embed = lambda x, eo=embedder_obj: eo.embed(x)
    return embed, embedder_obj.out_dim


class Embedder:
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.create_embedding_fn()

    def create_embedding_fn(self):
        embed_fns = []
        d = self.kwargs['input_dims']
        out_dim = 0
        if self.kwargs['include_input']:
            embed_fns.append(lambda x: x)
            out_dim += d

        max_freq = self.kwargs['max_freq_log2']
        N_freqs = self.kwargs['num_freqs']

        if self.kwargs['log_sampling']:
            freq_bands = 2. ** torch.linspace(0., max_freq, steps=N_freqs)
        else:
            freq_bands = torch.linspace(2. ** 0., 2. ** max_freq, steps=N_freqs)

        for freq in freq_bands:
            for p_fn in self.kwargs['periodic_fns']:
                embed_fns.append(lambda x, p_fn=p_fn, freq=freq: p_fn(x * freq))
                out_dim += d

        self.embed_fns = embed_fns
        self.out_dim = out_dim

    def embed(self, inputs):
        return torch.cat([fn(inputs) for fn in self.embed_fns], -1)


class DeformNetwork(nn.Module):
    def __init__(self, D=8, W=256, input_ch=3, output_ch=59, multires=10, is_blender=False, is_6dof=False):
        super(DeformNetwork, self).__init__()
        self.D = D  # 网络深度为8
        self.W = W  # 隐藏层维度为256
        self.input_ch = input_ch  # 输入通道3
        self.output_ch = output_ch  # 输出通道59
        self.t_multires = 6 if is_blender else 10  # 用于测试D-NeRF Dataset数据集时使用
        self.skips = [D // 2]  # 到skips进行一次残差连接

        self.embed_time_fn, time_input_ch = get_embedder(self.t_multires, 1)  # time的输入梯度函数和通道21
        self.embed_fn, xyz_input_ch = get_embedder(multires, 3)  # 位置的输入梯度函数和通道63
        self.input_ch = xyz_input_ch + time_input_ch  # 总的输入通道84

        if is_blender:
            # 调试D-NeRF Dataset数据集看是否能达到更好的效果
            self.time_out = 30

            self.timenet = nn.Sequential(
                nn.Linear(time_input_ch, 256), nn.ReLU(inplace=True),
                nn.Linear(256, self.time_out))

            '''
            self.linear = nn.ModuleList(...)：
            这行代码创建了一个nn.ModuleList实例，并将其赋值给self.linear。nn.ModuleList是PyTorch中的一个模块列表容器，
            它可以保存多个模块（如层），并且可以像普通列表一样进行索引和迭代。与普通列表不同的是，nn.ModuleList会将其包含的模块注册到网络中，
            这意味着这些模块的参数（权重和偏置）会在训练过程中被优化。
            [nn.Linear(xyz_input_ch + self.time_out, W)]：
            这是一个列表，其中包含一个nn.Linear层。这个线性层接受xyz_input_ch + self.time_out维的输入，并将其映射到W维的输出。
            这里xyz_input_ch和self.time_out是输入特征的维度，W是中间特征的维度。
            for i in range(D - 1)]：
            这是一个列表推导式，用于创建一个或多个nn.Linear层。D - 1表示要创建的线性层的数量（因为D是总层数，所以这里减去1）。
            nn.Linear(W, W) if i not in self.skips else nn.Linear(W + xyz_input_ch + self.time_out, W)：
            这是一个条件表达式，用于确定每个线性层的输入维度。
            如果当前层的索引i不在self.skips列表中，则该层的输入维度为W，输出维度也为W。
            如果当前层的索引i在self.skips列表中，则该层的输入维度为W + xyz_input_ch + self.time_out，输出维度为W。
            这意味着在某些特定的层（被标记为跳过层），输入维度会增加，可能用于融合额外的特征信息。
            综上所述，这段代码定义了一个包含D个线性层的网络结构，其中第一个层的输入维度是xyz_input_ch + self.time_out，输出维度是W。
            接下来的D - 1层中，除了索引在self.skips中的层以外，每层的输入和输出维度都是W。
            当索引i在self.skips中时，该层的输入维度会增加，以包含额外的特征信息。
            这种结构通常用于构建深度网络，其中self.skips可能用于实现类似于残差网络（ResNet）中的跳跃连接，
            允许网络在不同的层之间直接传递信息。
            '''
            self.linear = nn.ModuleList(
                [nn.Linear(xyz_input_ch + self.time_out, W)] + [
                    nn.Linear(W, W) if i not in self.skips else nn.Linear(W + xyz_input_ch + self.time_out, W)
                    for i in range(D - 1)]
            )

        else:
            self.linear = nn.ModuleList(
                [nn.Linear(self.input_ch, W)] + [
                    nn.Linear(W, W) if i not in self.skips else nn.Linear(W + self.input_ch, W)
                    for i in range(D - 1)]
            )

        self.is_blender = is_blender
        self.is_6dof = is_6dof

        if is_6dof:
            self.branch_w = nn.Linear(W, 3)
            self.branch_v = nn.Linear(W, 3)
        else:
            self.gaussian_warp = nn.Linear(W, 3)
        self.gaussian_rotation = nn.Linear(W, 4)
        self.gaussian_scaling = nn.Linear(W, 3)

    def forward(self, x, t):
        t_emb = self.embed_time_fn(t)
        if self.is_blender:
            t_emb = self.timenet(t_emb)  # better for D-NeRF Dataset
        x_emb = self.embed_fn(x)
        h = torch.cat([x_emb, t_emb], dim=-1)
        for i, l in enumerate(self.linear):
            h = self.linear[i](h)
            h = F.relu(h)
            if i in self.skips:
                h = torch.cat([x_emb, t_emb, h], -1)

        if self.is_6dof:
            w = self.branch_w(h)
            v = self.branch_v(h)
            theta = torch.norm(w, dim=-1, keepdim=True)
            w = w / theta + 1e-5
            v = v / theta + 1e-5
            screw_axis = torch.cat([w, v], dim=-1)
            d_xyz = exp_se3(screw_axis, theta)
        else:
            d_xyz = self.gaussian_warp(h)
        scaling = self.gaussian_scaling(h)
        rotation = self.gaussian_rotation(h)

        return d_xyz, rotation, scaling
