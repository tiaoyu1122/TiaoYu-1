import sys
import os
current_dir = os.path.dirname(os.path.abspath(__file__))  # 获取当前脚本所在的目录
parent_dir = os.path.dirname(current_dir)                 # 获取上一级目录
sys.path.append(parent_dir)                               # 将上一级目录添加到 sys.path 中，以便导入其他模块
from torch import nn                                # 导入PyTorch的nn模块，用于构建神经网络
from model_utils.Normalization import RMSNorm       # 导入RMSNorm类，用于实现RMS归一化操作
from model_utils.Attention import Attention         # 导入Attention类，用于实现自注意力机制
from model_utils.MoE import MOEFeedForward          # 导入MOEFeedForward类，用于实现MOE前馈神经网络
from model_config import TiaoyuConfig               # 导入TiaoyuConfig类，用于配置模型参数


# 定义解码器块类DecoderBlock，继承自nn.Module(配合"notebook/9-解码器模块.md"文档阅读)
class DecoderBlock(nn.Module):
    def __init__(self, 
                 layer_id: int,                  # 解码器块的编号
                 decoder_config: TiaoyuConfig):  # 模型配置参数
        # 调用父类的构造函数
        super().__init__()
        # 解码器块的编号
        self.layer_id = layer_id

        # (1) 初始化解码器块的参数
        self.embed_dim = decoder_config.embed_dim    # 模型的嵌入维度
        
        # (2) 初始化解码器块的组件之————注意力机制(配合"notebook/10-多头掩码自注意力机制.md"文档和"script/modeling/model_utils/Attention.py"代码阅读)
        self.Attention_norm = RMSNorm(dim=decoder_config.embed_dim, 
                                      epsilon=decoder_config.Norm_epsilon)    # 创建RMSNorm归一化实例，在执行注意力计算之前对输入进行归一化操作
        self.Attention = Attention(decoder_config=decoder_config)                            # 创建注意力机制的实例
        
        # (3) 初始化解码器块的组件之————前馈神经网络(配合"notebook/11-MOE前馈神经网络.md"文档和"script/modeling/model_utils/MoE.py"代码阅读)
        self.Feed_forward_norm = RMSNorm(dim=decoder_config.embed_dim, 
                                         epsilon=decoder_config.Norm_epsilon) # 创建RMSNorm归一化实例，在MOE前馈神经网络之前对输入进行归一化操作
        self.Moe_feed_forward = MOEFeedForward(decoder_config)                # 创建MOE前馈神经网络的实例
    
    # 定义前向传播方法
    def forward(self, x, freqs_cis, use_kv_cache=False, cache_k=None, cache_v=None):
        """
        Args:
            x (Tensor): 输入张量，形状为 [batch_size=32, seq_length=512, embed_dim=512]。
            freqs_cis (Tensor): 与输入张量对应的位置编码张量。
            use_kv_cache (bool): 是否使用键值缓存. 默认为False. 在生成过程中, 可以通过设置use_kv_cache=True来启用缓存.
            cache_k (torch.Tensor, optional): 本解码器层之前生成的 key 张量缓存. 通过复用这些缓存值, 可以避免重复计算, 从而加速生成过程.
            cache_v (torch.Tensor, optional): 本解码器层之前生成的 value 张量缓存. 通过复用这些缓存值, 可以避免重复计算, 从而加速生成过程.
        Returns:
            tuple: 包含输出张量和过去键值对的元组。
                - out (Tensor): 输出张量，形状为 [batch_size, seq_length, embed_dim]。
                - cache_k (torch.Tensor): 本次生成的 key 张量，用于缓存。
                - cache_v (torch.Tensor): 本次生成的 value 张量，用于缓存。
        """
        # (1) 调用注意力机制(输入x经过归一化处理), 返回值为h_attention(注意力输出)和cache_k、cache_v(键、值缓存)
        h_attention, cache_k, cache_v = self.Attention(
            x=self.Attention_norm(x),
            freqs_cis=freqs_cis,
            use_kv_cache=use_kv_cache, 
            cache_k=cache_k, 
            cache_v=cache_v
        )
        # (2) 将注意力机制输出与原始输入张量相加
        # 这是一种常见的设计，这种做法主要有以下4个方面的意义：
        # a. 保留原始信息：直接相加有助于保留输入中的原始信息，避免通过多层网络后丢失重要的特征。
        #                这样做可以确保模型不仅仅依赖于注意力机制生成的新特征表示，还能结合原始输入的特征，从而丰富最终的特征表达。
        # b. 简化模型结构：相比于直接将注意力机制的输出和原始输入拼接起来再通过一层线性变换来调整维度，
        #                简单地进行相加操作减少了参数数量和计算复杂度，有助于简化模型结构，同时也能一定程度上减少过拟合的风险。
        # c. 增强特征表达：通过注意力机制对某些特定部分的信息进行强调或抑制，然后将其结果与原始输入相结合，
        #                可以使模型更加关注于输入中最重要的部分，从而增强特征的表达能力，提高模型的性能。
        # d. 梯度流动：在训练深度神经网络时，保持良好的梯度流对于模型的有效训练至关重要。
        #            将注意力机制的输出与原始输入相加有助于维持一个较为稳定的梯度流动路径，有利于训练更深的网络结构。
        h = x + h_attention

        # (3) 调用MOE前馈神经网络(输入h经过归一化处理)，并将输出与h相加(设计原理同上)
        block_h = h + self.Moe_feed_forward(self.Feed_forward_norm(h))
        
        # (4) 返回输出block_h和键值缓存cache_k、cache_v
        return block_h, cache_k, cache_v