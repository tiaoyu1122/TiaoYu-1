import sys
import os
current_dir = os.path.dirname(os.path.abspath(__file__))  # 获取当前脚本所在的目录
parent_dir = os.path.dirname(current_dir)                 # 获取上一级目录
sys.path.append(parent_dir)                               # 将上一级目录添加到 sys.path 中，以便导入其他模块
import torch                       # 导入 PyTorch 库
from torch import nn               # 导入PyTorch的nn模块，用于构建神经网络
#导入 fairscale 库中模型并行相关模块
try:
    # fs_init 提供了一些诸如如设置进程组、获取当前设备的并行信息等的工具函数
    import fairscale.nn.model_parallel.initialize as fs_init
    # 获取当前模型并行组的数量。在分布式训练中，模型会被分割到多个 GPU 上运行。
    # 每个 GPU 负责模型的一部分计算，获取的即是这些 GPU 的总数。
    MODEL_PARALLEL_NUM = fs_init.get_model_parallel_world_size()
    from fairscale.nn.model_parallel.layers import (ColumnParallelLinear,   # 列并行线性层
                                                    RowParallelLinear)      # 行并行线性层
    USE_FAIRSCALE = True
except ImportError:
    # 如果没有安装 fairscale 库，则使用默认的单 GPU 配置
    MODEL_PARALLEL_NUM = 1
    USE_FAIRSCALE = False
from model_config import TiaoyuConfig              # 导入TiaoyuConfig类，用于配置模型参数
from model_utils.RoPE import apply_rotary_emb      # 导入自定义的旋转位置编码函数，用预先计算的角度来旋转输入的xq和xk
import torch.nn.functional as F                    # 导入PyTorch的函数库
import math                                        # 导入数学库


def repeat_kv(x: torch.Tensor, rep_times: int) -> torch.Tensor:
    """
    对输入的PyTorch张量(torch.Tensor)在特定的维度上进行复制(或“重复交织”)操作.
    Args:
    x: 待重复的张量
    rep_times: 重复的次数
    Returns:
    返回重复后的新张量
    """
    # 获取输入张量的形状(b, s, kv_head_num, head_dim)
    b, s, kv_head_num, head_dim = x.shape

    # 如果重复次数为1，则直接返回原始张量
    if rep_times == 1:
        return x
    # 否则，对张量进行重复操作，并返回新的张量
    return (
        x[:, :, :, None, :]
        .expand(b, s, kv_head_num, rep_times, head_dim)    # 对张量进行扩展，使得每个元素重复rep_times次
        .reshape(b, s, kv_head_num * rep_times, head_dim)  # 将扩展后的张量重新调整为(b, s, kv_head_num * rep_times, head_dim)的形状
    )


# 定义注意力机制类Attention，继承自nn.Module(配合"notebook/10-多头掩码自注意力机制.md"文档阅读)

class Attention(nn.Module):
    def __init__(self, 
                 decoder_config: TiaoyuConfig):
        # 调用父类的构造函数
        super().__init__()

        # (1) 初始化注意力机制所需的参数
        # a. 多头自注意力机制中键、值头的个数(与注意力权重的计算方式('MHA', 'MQA', 'GQA')相关)
        #    即Query的分组数量(组内共享键和值，Query分几组，就需要几个键和值头)
        self.kv_head_num = decoder_config.query_group_num
        # b. 单个 GPU 分配的多头自注意力机制中头的个数(即Query的头的个数)
        self.fairscale_q_head_num = decoder_config.head_num // MODEL_PARALLEL_NUM
        # c. 单个 GPU 分配的多头自注意力机制中键、值头的个数(即Key、Value的头的个数)
        self.fairscale_kv_head_num = self.kv_head_num // MODEL_PARALLEL_NUM
        # d. 要重复多少次Key、Value头才能与Query头数量对应
        self.rep_times = self.fairscale_q_head_num // self.fairscale_kv_head_num
        # e. 每个头的维度
        self.head_dim = decoder_config.embed_dim // decoder_config.head_num

        # (2) 初始化注意力机制所需的层
        # a. 初始化线性变换层
        if USE_FAIRSCALE:
            self.Wq = ColumnParallelLinear(input_size=decoder_config.embed_dim, 
                                           output_size=decoder_config.head_num * self.head_dim, 
                                           bias=False, 
                                           gather_output=False)    # 查询Query的线性变换层
            self.Wk = ColumnParallelLinear(input_size=decoder_config.embed_dim, 
                                           output_size=self.kv_head_num * self.head_dim, 
                                           bias=False, 
                                           gather_output=False)    # 键Key的线性变换层
            self.Wv = ColumnParallelLinear(input_size=decoder_config.embed_dim, 
                                           output_size=self.kv_head_num * self.head_dim, 
                                           bias=False, 
                                           gather_output=False)    # 值Value的线性变换层
            self.Wo = RowParallelLinear(input_size=decoder_config.head_num * self.head_dim, 
                                        output_size=decoder_config.embed_dim, 
                                        bias=False, 
                                        input_is_parallel=True)    # 输出线性变换层
        else:
            self.Wq = nn.Linear(in_features=decoder_config.embed_dim, 
                                out_features=decoder_config.head_num * self.head_dim, 
                                bias=False)                        # 查询Query的线性变换层
            self.Wk = nn.Linear(in_features=decoder_config.embed_dim, 
                                out_features=self.kv_head_num * self.head_dim, 
                                bias=False)                        # 键Key的线性变换层
            self.Wv = nn.Linear(in_features=decoder_config.embed_dim, 
                                out_features=self.kv_head_num * self.head_dim, 
                                bias=False)                        # 值Value的线性变换层
            self.Wo = nn.Linear(in_features=decoder_config.head_num * self.head_dim, 
                                out_features=decoder_config.embed_dim, 
                                bias=False)                        # 输出线性变换层
        # b. 初始化dropout概率及Dropout层(dropout概率将在F.scaled_dot_product_attention缩放点积注意力计算函数中使用)
        self.dropout_p = decoder_config.Dropout_p                            # 全局dropout概率
        self.Dropout_attention = nn.Dropout(decoder_config.Dropout_p)        # 注意力分数计算使用的Dropout层
        self.Dropout_output = nn.Dropout(decoder_config.Dropout_p)           # 注意力结果计算使用的Dropout层
        
        # 提前计算 sqrt(self.head_dim)
        self.sqrt_head_dim = math.sqrt(self.head_dim)
        
        # (3) 初始化是否使用快速点积注意力函数
        self.flash = hasattr(torch.nn.functional, 'scaled_dot_product_attention')  # 是否可以获取PyTorch 2.0中的scaled_dot_product_attention函数
        if self.flash:
            print("Using scaled_dot_product_attention!")
        else:
            print("Using tiaoyu_dot_product_attention!")
        
        # (4) 初始化掩码矩阵，在自注意力机制中屏蔽无效或不需要的位置(依靠 -inf 来实现屏蔽)
        # a. 生成一个形状为(1, 1, decoder_config.max_seq_len, decoder_config.max_seq_len)的全为-inf的张量
        mask = torch.full((1, 1, decoder_config.max_seq_len, decoder_config.max_seq_len), float("-inf")) 
        # b. 对上一步生成的张量应用了上三角函数 torch.triu，保留主对角线以上的部分(不包括主对角线)，并将其他位置的值设置为 0
        #    这用于确保注意力是因果的（即，每个位置只能关注到其前面的位置，后面的是 -inf 被屏蔽掉了）
        mask = torch.triu(mask, diagonal=1)
        # c. 生成的掩码张量注册为模型的一个缓冲区
        #    在 PyTorch 中，缓冲区是一种特殊的张量，它不是模型的可训练参数，但与模型相关联，会在模型的前向传播中使用。
        #    这里的掩码张量 mask 是一个固定值，不需要参与梯度计算，因此适合注册为缓冲区。
        #    如果 persistent=False，则 mask 不会被保存到模型的 state_dict 中，节省存储空间。
        self.register_buffer("mask", mask, persistent=False) 

    # 前向传播函数
    def forward(self,
                x: torch.Tensor,         # 输入张量
                freqs_cis: torch.Tensor, # 与输入张量对应的位置编码张量
                use_kv_cache=False,      # 一个布尔值，表示是否使用缓存
                cache_k=None,            # 所属解码器层之前生成的 key 张量缓存
                cache_v=None):           # 所属解码器层之前生成的 value 张量缓存
        
        # (1) 获取输入张量的形状(批次大小b，序列长度s)
        b, s, _ = x.shape
        
        # (2) 计算查询、键和值的线性变换结果，并进行形状重塑
        # a. 分别计算查询、键和值的线性变换结果
        xq, xk, xv = self.Wq(x), self.Wk(x), self.Wv(x)
        # b. 分别重塑查询、键和值的线性变换结果
        xq = xq.view(b, s, self.fairscale_q_head_num, self.head_dim)
        xk = xk.view(b, s, self.fairscale_kv_head_num, self.head_dim)
        xv = xv.view(b, s, self.fairscale_kv_head_num, self.head_dim)
        
        # (3) 对 Query 和 Key 应用旋转位置编码
        xq, xk = apply_rotary_emb(xq=xq, xk=xk, freqs_cis=freqs_cis)
        
        # (4) 使用和更新 Key 和 Value 缓存
        # a. 使用缓存中的 Key 和 Value，与当前输入的Key和Value进行拼接
        if cache_k is not None:
            xk = torch.cat([cache_k, xk], dim=1)
            xv = torch.cat([cache_v, xv], dim=1)
        # b. 更新 Key 和 Value 缓存
        cache_k = xk if use_kv_cache else None
        cache_v = xv if use_kv_cache else None
        
        # (5) 重塑 Query, Key, Value 张量的形状
        xq, xk, xv = (
            xq.transpose(1, 2),                                             # 将查询张量的第 1 维和第 2 维进行交换(转置)
            repeat_kv(x=xk, rep_times=self.rep_times).transpose(1, 2), # 重复键张量并转置，使其与查询张量具有相同的形状
            repeat_kv(x=xv, rep_times=self.rep_times).transpose(1, 2)  # 重复值张量并转置，使其与查询张量具有相同的形状
        )

        # (6) 计算注意力分数
        use_flash = self.flash and s != 1
        if use_flash:
            # 使用PyTorch 2.0中的注意力计算函数scaled_dot_product_attention
            dropout_p = self.dropout_p if self.training else 0.0
            output = F.scaled_dot_product_attention(
                query=xq,            # 查询张量
                key=xk,              # 键张量     
                value=xv,            # 值张量
                attn_mask=None,      # 注意力掩码(Attention Mask)，用于控制哪些位置可以参与注意力计算
                dropout_p=dropout_p, # dropout 概率，用于防止过拟合
                is_causal=True       # 是否是因果注意力(Causal Attention)，如果为True，则只计算当前位置之前的注意力
            )
        else:
            # 调用自定义的注意力计算代码
            # a. xq @ xk.transpose(-2, -1): 计算查询向量和键向量的点积。
            #    xk.transpose(-2, -1) 将键向量的最后两个维度进行转置，以便与查询向量进行矩阵乘法(@表示矩阵乘法)
            # b. / math.sqrt(self.head_dim): 为了防止点积结果过大，导致softmax函数梯度消失，
            #    将点积结果除以键向量维度的平方根(self.head_dim 是每个“头”的维度)。
            # c. + self.mask[:, :, :s, :s]: 将掩码加到分数上，以便在需要的位置禁止注意力。
            #    通过将掩码值设置为负无穷来实现，这样softmax在这些位置上的输出将接近0。
            # scores = (xq @ xk.transpose(-2, -1)) / math.sqrt(self.head_dim) + self.mask[:, :, :s, :s]
            scores = (xq @ xk.transpose(-2, -1)) / self.sqrt_head_dim + self.mask[:, :, :s, :s]
            # d. F.softmax(scores.float(), dim=-1): 对注意力分数应用softmax函数，使它们成为概率分布。
            #    dim=-1 表示在最后一个维度上应用softmax。
            # e. .type_as(xq): 确保softmax后的分数与查询向量的数据类型相同。
            # f. Dropout_attention(...): 对注意力分数应用Dropout层，以减少过拟合风险。
            scores = self.Dropout_attention(F.softmax(scores.float(), dim=-1).type_as(xq))
            # g. scores @ xv: 将注意力分数与值向量相乘，得到最终的注意力结果。
            output = scores @ xv

        # 将输出张量重塑为(b, s, -1)的形状
        output = output.transpose(1, 2).reshape(b, s, -1)
        
        # (7) 应用线性变换及Dropout
        output = self.Dropout_output(self.Wo(output))

        # 返回注意力结果及更新后的Key和Value缓存
        return output, cache_k, cache_v
        