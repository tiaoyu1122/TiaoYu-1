import torch           # 导入PyTorch库
from torch import nn   # 导入PyTorch的神经网络模块

"""
这里定义了两个类：LayerNorm和RMSNorm.
(1) LayerNorm类实现了层归一化操作.
(2) RMSNorm类实现了RMS归一化操作.
实际我们在模型中使用的是RMSNorm. LayerNorm仅作为演示使用, 感兴趣的同学可以将模型中的RMSNorm替换为LayerNorm进行对比实验.
"""

# 定义了一个名为LayerNorm的类，它是torch.nn.Module的子类，用于实现层归一化操作
class LayerNorm(nn.Module):
    def __init__(self, 
                 dim: int, 
                 epsilon: float = 1e-6):
        """
        LayerNorm 初始化方法.
        Args:
            dim (int): 嵌入维度.
            epsilon (float): 一个小的正数，用于防止分母为零.
        """
        # 调用父类的构造函数
        super().__init__()
        # 初始化一个常量epsilon
        self.epsilon = epsilon
        # 创建一个可学习的参数gamma (即层归一化方法介绍中的缩放因子)，其形状由dim指定，并初始化为全1。这个参数会在训练过程中被优化。
        self.gamma = nn.Parameter(torch.ones(dim))
        # 创建一个可学习的参数beta (即层归一化方法介绍中的偏移量)，其形状由dim指定，并初始化为全0。这个参数会在训练过程中被优化。
        self.beta = nn.Parameter(torch.zeros(dim))
    
    # 定义前向传播方法
    def forward(self, x):
        """
        对输入 x 进行标准化处理，乘以缩放因子 gamma, 加上偏移量 beta.
        Args:
            x (torch.Tensor): 输入张量，形状为 (b, s, d)，其中 b 是批量大小，s 是序列长度，d 是嵌入维度.
        Returns:
            torch.Tensor: 层归一化处理后的张量，形状与输入 x 相同.
        """
        # 计算输入张量的均值和方差，分别作为mean和var
        mean = x.mean(-1, keepdim=True)
        var = x.var(-1, unbiased=False, keepdim=True)
        # 对输入张量进行标准化处理，并乘以缩放因子 gamma, 加上偏移量 beta
        x_norm = (x.float() - mean) / torch.sqrt(var + self.epsilon)
        # 将标准化后的张量缩放因子gamma相乘，再加上偏移量beta，得到最终的输出张量
        return self.gamma * x_norm + self.beta
        

# 定义了一个名为RMSNorm的类，它是torch.nn.Module的子类，用于实现 RMS 归一化操作
class RMSNorm(nn.Module):
    def __init__(self, 
                 dim: int, 
                 epsilon: float = 1e-6):
        """
        RMSNorm 初始化方法.
        Args:
            dim (int): 嵌入维度.
            epsilon (float): 一个小的正数，用于防止分母为零.
        """
        # 调用父类的构造函数
        super().__init__()
        # 初始化一个常量epsilon
        self.epsilon = epsilon
        # 创建一个可学习的参数gamma (即RMS归一化方法介绍中的缩放因子)，其形状由dim指定，并初始化为全1。这个参数会在训练过程中被优化。
        self.gamma = nn.Parameter(torch.ones(dim)) 
    
    # 定义前向传播方法
    def forward(self, x):
        """
        对输入 x 进行标准化处理，并乘以缩放因子 gamma.
        Args:
            x (torch.Tensor): 输入张量，形状为 (b, s, d)，其中 b 是批量大小，s 是序列长度，d 是嵌入维度.
        Returns:
            torch.Tensor: RMS 归一化处理后的张量，形状与输入 x 相同.
        """
        # 计算输入张量的RMS倒数值，并将其与epsilon相加以防止分母为零(这里先加了epsilon，与文档介绍中的公式有差别，但实际效果是一致的)
        # a. torch.square(x): 对输入张量 x 的每个元素平方
        # b. mean(-1, keepdim=True) 沿着最后一个维度取平均值，keepdim=True保留原来的张量形状，即保持(b, s, 1)
        # c. + self.epsilon 加上一个很小的 epsilon，以防止分母为零
        # d. torch.rsqrt() 表示计算输入张量的"倒数平方根"
        rms_reciprocal = torch.rsqrt(torch.square(x).mean(-1, keepdim=True) + self.epsilon)
        # 将输入张量与RMS值相乘，然后再乘以缩放因子 gamma，得到标准化后的张量
        # a. x.float() 将输入张量转换为浮点类型
        # b. rms_reciprocal 是 RMS 的倒数，所以这里是相乘，而不是相除
        norm_x = x * rms_reciprocal 
        # 将标准化后的张量与缩放因子 gamma 相乘，得到最终的输出张量
        return self.gamma * norm_x


if __name__ == "__main__":
    # 设置随机数种子
    torch.manual_seed(12)
    # 创建一个张量x，形状为(2, 3, 5)
    x = torch.arange(24).reshape(2, 3, 4).float()
    print(f'x: {x}')
    # 打印结果如下:
    # x: tensor([[[ 0.,  1.,  2.,  3.],
    #             [ 4.,  5.,  6.,  7.],
    #             [ 8.,  9., 10., 11.]],

    #            [[12., 13., 14., 15.],
    #             [16., 17., 18., 19.],
    #             [20., 21., 22., 23.]]])

    # (1) RMS 归一化结果
    # 创建一个RMSNorm实例, 并设置嵌入维度为x.shape[-1]
    rmsnorm = RMSNorm(dim=x.shape[-1], epsilon=1e-5)
    # 将x输入到RMSNorm实例中进行归一化处理，得到归一化后的张量rms_norm_x
    rms_norm_x = rmsnorm(x)
    print(f'RMS归一化结果: {rms_norm_x}')
    # 打印结果如下:
    # RMS归一化结果: tensor([[[0.0000, 0.5345, 1.0690, 1.6036],
    #                       [0.7127, 0.8909, 1.0690, 1.2472],
    #                       [0.8363, 0.9409, 1.0454, 1.1500]],

    #                      [[0.8859, 0.9597, 1.0335, 1.1073],
    #                       [0.9124, 0.9695, 1.0265, 1.0835],
    #                       [0.9290, 0.9754, 1.0219, 1.0683]]], grad_fn=<MulBackward0>)

    # (2) 层归一化结果
    # 创建一个LayerNorm实例, 并设置嵌入维度为x.shape[-1]
    layernorm = LayerNorm(dim=x.shape[-1], epsilon=1e-5)
    # 将x输入到LayerNorm实例中进行归一化处理，得到归一化后的张量layer_norm_x
    layer_norm_x = layernorm(x)
    print(f'层归一化结果: {layer_norm_x}')
    # 打印结果如下:
    # 层归一化结果: tensor([[[-1.3416, -0.4472,  0.4472,  1.3416],
    #                      [-1.3416, -0.4472,  0.4472,  1.3416],
    #                      [-1.3416, -0.4472,  0.4472,  1.3416]],

    #                     [[-1.3416, -0.4472,  0.4472,  1.3416],
    #                      [-1.3416, -0.4472,  0.4472,  1.3416],
    #                      [-1.3416, -0.4472,  0.4472,  1.3416]]], grad_fn=<AddBackward0>)
    