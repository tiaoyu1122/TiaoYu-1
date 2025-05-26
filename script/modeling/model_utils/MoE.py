import sys
import os
current_dir = os.path.dirname(os.path.abspath(__file__))  # 获取当前脚本所在的目录
parent_dir = os.path.dirname(current_dir)                 # 获取上一级目录
sys.path.append(parent_dir)                               # 将上一级目录添加到 sys.path 中，以便导入其他模块
import torch                           # 导入PyTorch
from torch import nn                   # 导入PyTorch的神经网络模块
from model_config import TiaoyuConfig  # 导入自定义的模型配置类
import math                            # 导入数学模块
import torch.nn.functional as F        # 导入PyTorch的函数库
import torch.nn.init as init          # 导入PyTorch的初始化模块
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
from transformers.activations import ACT2FN


# 定义一个专家(基础的前馈神经网络 FeedForward)，在后续 MOEFeedForward 中使用
class FeedForward(nn.Module):
    # 初始化方法
    def __init__(self, 
                 decoder_config: TiaoyuConfig):
        # 调用父类nn.Module的初始化方法
        super().__init__()
        # 计算前馈神经网络层中隐藏层的维度，并定义 3 个线性层
        hidden_dim = int(decoder_config.embed_dim * decoder_config.ffn_dim_multiplier)
        if USE_FAIRSCALE:
            self.Linear_1 = ColumnParallelLinear(decoder_config.embed_dim, hidden_dim, bias=False, gather_output=False)
            self.Linear_2 = RowParallelLinear(hidden_dim, decoder_config.embed_dim, bias=False, input_is_parallel=True)
            self.Linear_3 = ColumnParallelLinear(decoder_config.embed_dim, hidden_dim, bias=False, gather_output=False)
        else:
            self.Linear_1 = nn.Linear(decoder_config.embed_dim, hidden_dim, bias=False)
            self.Linear_2 = nn.Linear(hidden_dim, decoder_config.embed_dim, bias=False)
            self.Linear_3 = nn.Linear(decoder_config.embed_dim, hidden_dim, bias=False)
        # 定义一个dropout层    
        self.Dropout = nn.Dropout(decoder_config.Dropout_p)
        # 定义一个激活函数层
        self.SiLU = ACT2FN['silu']

    # 前向传播函数
    def forward(self, x):
        # 假设输入的x是一个形状为(b, s, d=embed_dim)的张量
        # a. self.Linear_1(x) 计算，得到一个新的张量，维度为 (b, s, hidden_dim)
        # b. self.Linear_3(x) 计算，得到一个新的张量，维度为 (b, s, hidden_dim)
        # c. F.silu(self.Linear_1(x)) 计算，即在 a 的结果基础上应用 F.silu()，得到一个新的张量，维度为 (b, s, hidden_dim)
        # d. F.silu(self.Linear_1(x)) * self.Linear_3(x) 计算，即 d 的结果与 b 的结果逐元素相乘，得到一个新的张量，维度为 (b, s, hidden_dim)
        #    这种乘法操作是一种 SwiGLU 的神经网络单元：将同一个输入分别发送到两个不同的分支，一个分支通过线性变换后应用 silu 激活函数，
        #    另一个分支通过线性变换直接输出。最后，将两个分支的结果逐元素相乘。这种操作的优点有：
        #      i. 非线性与门控机制，增强的表达能力：通过结合激活函数和门控机制来增强模型的表达能力。相比于简单的线性变换，这种组合允许网络学习更复杂的特征映射，并能动态地控制信息流，从而捕捉输入数据中的细微差异。
        #      ii. 平滑的梯度特性，改进的学习动态：silu(也称为Swish)是一种平滑且连续可导的激活函数，相较于ReLU等激活函数，它拥有更好的梯度特性，有助于缓解深度网络中常见的梯度消失或爆炸问题，进而改善训练过程中的学习动态。
        #      iii. 提升模型性能：研究表明，在许多自然语言处理任务中，使用SwiGLU的模型（如PaLM、LLaMA等）表现出了优于使用传统线性层模型的性能。这主要是因为SwiGLU能够更好地捕捉序列数据中的长距离依赖关系和复杂模式。
        #      iv. 高效的计算结构：尽管SwiGLU引入了额外的乘法操作，但其整体计算架构仍然是轻量级的，特别是在现代硬件（如GPU/TPU）上执行时，这种设计非常高效，能够在不显著增加计算成本的情况下提高模型的表现力。
        # e. self.Linear_2(F.silu(self.Linear_1(x)) * self.Linear_3(x)) 计算，即 d 的结果输入到 self.Linear_2() 中，得到一个新的张量，维度为 (b, s, d)
        # f. self.Dropout(self.Linear_2(F.silu(self.Linear_1(x)) * self.Linear_3(x))) 计算，即 e 的结果经过 dropout 操作，得到一个新的张量，维度为 (b, s, d)
        return self.Dropout(self.Linear_2(self.SiLU(self.Linear_1(x)) * self.Linear_3(x)))


# 定义一个门控模块 MOEGate，用于选择使用哪些专家，以及如何分配权重
class MOEGate(nn.Module):
    # 初始化方法
    def __init__(self, 
                 decoder_config: TiaoyuConfig):
        # 调用父类nn.Module的初始化方法
        super().__init__() 
        
        # 初始化专家数量和使用的专家数量、辅助损失权重
        self.expert_num = decoder_config.expert_num
        self.expert_use = decoder_config.expert_use
        self.aux_loss_lambda = decoder_config.aux_loss_lambda
        
        # 定义一个线性层，输入维度为embed_dim，输出维度为expert_num
        self.Linear = nn.Linear(decoder_config.embed_dim, self.expert_num, bias=False)

    def forward(self, h):

        # (1) 将输入张量展平为二维张量，维度为 (b * s, d)
        b, s, d = h.shape
        h = h.view(-1, d)
        
        # (2) 计算每个token对每个专家的得分，得到一个形状为 (b * s, expert_num) 的张量
        scores = F.softmax(self.Linear(h), dim=-1)
        
        # (3) 选择得分最高的 expert_use 个专家，得到的 expert_scores, expert_ids 形状为 (b * s, expert_use)
        expert_scores, expert_ids = torch.topk(scores, k=self.expert_use, dim=-1, sorted=False)
        # 计算被选中的专家权重(确保权重和为1)，形状为 (b * s, expert_use)
        expert_weight = expert_scores / expert_scores.sum(dim=-1, keepdim=True)
        
        if self.training and self.aux_loss_lambda > 0.0:
            # 计算辅助损失
            aux_loss = self._compute_load_balance_loss(scores, b, s)
        else:
            # 如果不是训练模式，则辅助损失为0
            aux_loss = 0
            
        return expert_ids, expert_weight, aux_loss
    
    # 计算负载均衡损失的方法
    # 这个损失会惩罚那些被过度使用的专家，鼓励更均匀的负载分布，目标是让每个专家处理的样本数量尽可能均匀
    def _compute_load_balance_loss(self, scores, b, s):
        """计算基于L2范数的负载均衡损失"""
        # 将scores从(b*s, expert_num)变为(b, s, expert_num)，便于计算每个序列的损失
        scores = scores.view(b, s, -1)
        # 计算每个序列的专家概率分布（沿sequence维度平均）
        seq_probs = scores.mean(dim=1)  # shape: [b, expert_num]
        # 计算每个序列的L2损失（平方和）
        seq_losses = torch.sum(seq_probs ** 2, dim=-1)  # shape: [b]
        # 对batch求平均并缩放
        return seq_losses.mean() * self.expert_num * self.aux_loss_lambda


# 定义一个门控专家前馈神经网络 MOEFeedForward，包含一个共享专家前馈神经网络、多个专家的混合专家前馈神经网络、一个门控模块
class MOEFeedForward(nn.Module):
    # 初始化方法
    def __init__(self, 
                 decoder_config: TiaoyuConfig):
        # 调用父类初始化方法
        super().__init__()
        # 初始化模型配置超参数
        self.decoder_config = decoder_config
        # 初始化包含 1 个专家的共享的专家前馈神经网络
        self.Shared_expert = FeedForward(decoder_config)
        # 初始化 expert_num 个专家的混合专家前馈神经网络
        self.Mixture_experts = nn.ModuleList([FeedForward(decoder_config) for _ in range(decoder_config.expert_num)])
        # 初始化门控模块，用来选择专家和计算权重、输出辅助损失
        self.Gate = MOEGate(decoder_config)
        # 初始化一个空的辅助损失变量
        self.aux_loss = None       

    # 前向传播函数
    def forward(self, h):
        
        # (1) 获取输入数据形状 (批次大小, 序列长度, 嵌入维度)
        b, s, d = h.shape

        # (2) 计算共享专家的输出，得到的 shared_y 形状为 (b, s, d)
        shared_y = self.Shared_expert(h)

        # (3) 使用门控机制选择专家，输出结果包括：
        #     - expert_ids: 形状为 (b * s, expert_use)，表示每个token选择的专家索引
        #     - expert_weight: 形状为 (b * s, expert_use)，表示每个token选择的专家的权重
        #     - aux_loss: 辅助损失，用于训练时的优化过程
        expert_ids, expert_weight, aux_loss = self.Gate(h)
        
        # (4) 数据变形
        # 将输入数据展平，形状变为 (b * s, d)
        h = h.view(-1, d)

        if self.training: # 训练模式下
            # a. 将选择专家的索引展平到1维，形状变为 (b * s * expert_use) 的1维张量
            expert_ids = expert_ids.view(-1)
            # b. 在 0 维上将输入数据重复 expert_use 次，得到 (b * s * expert_use, d) 的张量
            h = h.repeat_interleave(self.decoder_config.expert_use, dim=0)
            # c. 创建一个空的张量 y，形状与 h 相同，为 (b * s * expert_use, d)，用于存储每个专家的输出
            y = torch.empty_like(h, dtype=torch.float16)
            # d. 遍历每个专家，对输入数据应用对应的专家前馈神经网络，并将结果存储到 y 中
            for i, expert in enumerate(self.Mixture_experts):
                # 对第 i 个专家，选择 expert_ids 中等于 i 的索引，将对应的输入数据传入专家前馈神经网络，将结果存储到 y 中
                y[expert_ids == i] = expert(h[expert_ids == i]).to(y.dtype)  # 确保类型一致
            # e. i. y.view(*expert_weight.shape, -1) 将 y 的形状从 (b * s * expert_use, d) 变形为 (b * s, expert_use, d)
            #    ii. expert_weight.unsqueeze(-1) 对 expert_weight 添加一个维度，变形为 (b * s, expert_use, 1)
            #    iii. y.view(*expert_weight.shape, -1) * expert_weight.unsqueeze(-1) 进行乘法操作，将每个专家的输出乘以对应的权重，并在专家维度上进行加权求和，得到的输出形状为 (b * s, d)
            y = (y.view(*expert_weight.shape, -1) * expert_weight.unsqueeze(-1)).sum(dim=1)
            # f. 恢复原始形状
            y = y.view(b, s, d)
        else:
            # 推理模式下，只选择最优专家
            y = self._infer(h, expert_ids, expert_weight).view(b, s, d)
        
        # (5) 将结果与共享专家层的结果相加
        y = y + shared_y

        # (6) 保存辅助损失
        self.aux_loss = aux_loss

        # (7) 返回最终输出
        return y

    # 推理模式下前向传播计算
    @torch.no_grad()
    def _infer(self, h, expert_ids, expert_weight):
        
        # (1) 创建输出张量，形状与 h 相同，为(b * s, d)，初始值全为0，用于存储输出结果
        y = torch.zeros_like(h)

        # (2) 对 h 中的 b * s 个 token 对应的嵌入向量依次进行处理，并将结果累加到 y 中
        for i in range(h.size(0)):
            # a. 获取当前 token 选择的专家ID和权重
            this_experts = expert_ids[i]    # expert_ids 的形状为 (b * s, expert_use)，this_experts 的形状为 (1, expert_use)
            this_weights = expert_weight[i] # expert_weight 的形状为 (b * s, expert_use)，this_weights 的形状为 (1, expert_use)

            # b. 计算加权输出，对每个专家，将输入数据传入对应的专家前馈神经网络，将结果乘以对应的权重，并在专家维度上进行加权求和
            for j in range(self.decoder_config.expert_use):
                # i. 获取当前专家的索引
                this_expert_id = this_experts[j].item()
                # ii. 将当前 token 对应的嵌入向量数据输入到对应的专家前馈神经网络中，得到专家的输出，维度为 (1, d)，循环 b * s * expert_use 次，共得到 b * s * expert_use个输出
                this_expert_out = self.Mixture_experts[this_expert_id](h[i].unsqueeze(0))
                # iii. 同一个 token 的 expert_use 个不同专家的输出会被加权，并累加到 y 中，结果维度为 (b * s, d)。 
                y[i] += this_expert_out.squeeze(0) * this_weights[j]
        
        return y


if __name__ == "__main__":
    b, s, d = 2, 3, 16
    h = torch.randn(b, s, d)

    config = TiaoyuConfig(max_batch_size=b, max_seq_len=s, embed_dim=d)
    
    model = MOEFeedForward(config)
    model.eval()
    
    y = model(h)
    