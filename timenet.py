import torch
from torch import nn
"""
    将时间变量通过Embedder进行升维
"""

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
        '''torch.linspace(0., 5, steps=7)
            生成的张量是：[0., 0.85, 1.71, 2.57, 3.43, 4.29, 5.]（这些值是0到5之间6个等间隔的点，加上端点5，总共7个点）。
            对上述张量中的每个元素取2的幂，得到freq_bands：
            [2^0, 2^0.85, 2^1.71, 2^2.57, 2^3.43, 2^4.29, 2^5]。
            即tensor([ 1.0000,  1.7818,  3.1748,  5.6569, 10.0794, 17.9594, 32.0000])'''

        for freq in freq_bands:
            for p_fn in self.kwargs['periodic_fns']:
                embed_fns.append(lambda x, p_fn=p_fn, freq=freq: p_fn(x * freq))
                out_dim += d
        '''x：这是lambda函数的第一个参数，它代表输入值。
        p_fn=p_fn：这是lambda函数的第二个参数，它有一个默认值p_fn。这意味着当你调用这个lambda函数时，你可以不传递p_fn参数，它将使用外部作用域中的p_fn变量的值。
        freq=freq：这是lambda函数的第三个参数，它有一个默认值freq。同样，这意味着你可以不传递freq参数，它将使用外部作用域中的freq变量的值。
        p_fn(x * freq)：这是lambda函数的主体，它返回对x和freq进行乘法操作后的结果应用p_fn函数的结果。'''
        self.embed_fns = embed_fns
        self.out_dim = out_dim

    def embed(self, inputs):
        for fn in self.embed_fns:
            fn(inputs)
        return torch.cat([fn(inputs) for fn in self.embed_fns], -1)

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

# 创建Embedder对象和嵌入函数
t_multires = 6

embed_time_fn, time_input_ch = get_embedder(t_multires, 1)

# 创建一个输入张量，这里假设输入是一个长度为1的张量，值为0.5
input_tensor = torch.tensor([[0.5]])

# 使用嵌入函数来生成嵌入后的张量
embedded_tensor = embed_time_fn(input_tensor)


# 定义timenet网络，确保输入维度与embedded_tensor的最后一个维度相匹配
timenet = nn.Sequential(
    nn.Linear(time_input_ch, 256),  # 将3改为time_input_ch
    nn.ReLU(inplace=True),
    nn.Linear(256, 30)
)

# 将嵌入后的张量传递给timenet
output = timenet(embedded_tensor)

print(output)