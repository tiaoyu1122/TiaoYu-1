import sys
import os
current_dir = os.path.dirname(os.path.abspath(__file__))  # 获取当前脚本所在的目录
parent_dir = os.path.dirname(current_dir)                 # 获取上一级目录
sys.path.append(parent_dir)                               # 将上一级目录添加到 sys.path 中，以便导入其他模块
from transformers import PreTrainedModel                    # 导入预训练模型基类(定义TIAOYU模型时需要继承该类)
import torch                                                # 导入PyTorch
from torch import nn                                        # 导入PyTorch的神经网络模块
import torch.nn.functional as F                             # 导入PyTorch的函数库
# 导入3个类型注解，它们有助于提高代码的可读性和可维护性，还可以帮助一些工具（如类型检查器）更好地理解代码的预期行为
from typing import (Optional, # Optional类型注解用于表示一个变量、函数参数或返回值可以是某个类型，也可以是None
                    List)     # List类型注解用于表示一个列表类型
from modeling.model_config import TiaoyuConfig              # 导入自定义的模型配置类
from modeling.model_utils.Normalization import RMSNorm               # 导入自定义的归一化层类
from modeling.model_utils.DecoderBlock import DecoderBlock           # 导入自定义的解码器块类
from modeling.model_utils.RoPE import precompute_freqs_cis           # 导入自定义的旋转位置编码函数，用于预先计算位置编码的复数形式笛卡尔坐标
from modeling.model_utils.Output import TiaoyuCausalLMOutputWithPast # 导入自定义的模型输出类
#导入 fairscale 库中模型并行相关模块
try:
    from fairscale.nn.model_parallel.layers import (VocabParallelEmbedding, # 并行词嵌入层
                                                    ColumnParallelLinear)   # 行并行线性层
    USE_FAIRSCALE = True
except ImportError:
    print("Warning: 没有找到 fairscale 库, 使用默认设置! 或者您可以终止运行, 并安装 fairscale 库后重新尝试!")
    USE_FAIRSCALE = False


class TIAOYU(PreTrainedModel):
    """
    定义TIAOYU模型, 继承自PreTrainedModel。
    
    PreTrainedModel 是 Hugging Face 的 transformers 库中的一个核心基类, 它:
     - 提供了统一的接口设计;
     - 无缝集成预训练模型;
     - 自动支持多种功能(如 save_pretrained, from_pretrained 和 .to(device) 等方法, 以及序列化与反序列化等)
    
    TIAOYU类下实现了以下函数:
     - 构造函数 __init__(), 用于初始化类的实例;
     - 定义模型的前向传播函数 forward(), 用于:
       + 定义前向传播逻辑: 当我们将输入数据传递给模型时(如 output = model(input))，实际上会调用模型的 forward 方法;
       + 支持自定义操作: 当我们自定义操作时, 可以通过重写forward方法来实现自定义逻辑;
       + 结合自动求导: forward 函数中的操作会被 PyTorch 的自动求导系统记录下来，从而支持反向传播和梯度计算.
     - 定义模型的补全函数 complete() 及 _stream_complete(), 用于训练后的模型使用, 即完成输入文本的补全操作.
    """
    
    def __init__(self, tiaoyu_config: TiaoyuConfig = None):
        """
        初始化TIAOYU模型
        Args:
            tiaoyu_config (ModelConfig): 模型配置超参数
        """
        super().__init__(tiaoyu_config)        # 调用父类的构造函数
        
        # 1. 初始化模型配置超参数
        self.tiaoyu_config = tiaoyu_config     

        # 2. 初始化模型各层
        # (1) 嵌入层
        if USE_FAIRSCALE:
            self.Embedding = VocabParallelEmbedding(num_embeddings=tiaoyu_config.vocab_size, # 词汇表大小
                                                    embedding_dim=tiaoyu_config.embed_dim)   # 每个嵌入向量的维度
        else:
            self.Embedding = nn.Embedding(num_embeddings=tiaoyu_config.vocab_size, # 词汇表大小
                                          embedding_dim=tiaoyu_config.embed_dim)   # 每个嵌入向量的维度
        # (2) Dropout层
        self.Dropout = nn.Dropout(p=tiaoyu_config.Dropout_p)   # p定义了每个神经元被丢弃的概率
        # (3) 预计算旋转位置编码矩阵(因为不是torch.nn.Parameter 对象, 所以不会在反向传播时参与优化)
        self.freqs_cis = precompute_freqs_cis(dim=tiaoyu_config.embed_dim // tiaoyu_config.head_num,  # 输入维度大小, 可以理解成embedding的维度
                                              end=tiaoyu_config.max_seq_len,                          # 结束位置索引, 为序列的最大长度
                                              theta=tiaoyu_config.RoPE_theta)                         # 旋转位置编码的参数theta
        # (4) 解码器层列表(每个解码器层又由自注意力层和前馈神经网络层组成)
        # nn.ModuleList 是一个容器, 用于存储一系列模型层, 它可以方便地添加、删除和访问模型层.
        # 这里创建了一个包含tiaoyu_config.layer_num个DecoderBlock的列表
        self.Blocks = nn.ModuleList()
        for layer_id in range(tiaoyu_config.layer_num):
            self.Blocks.append(DecoderBlock(layer_id=layer_id, decoder_config=tiaoyu_config))
        # (5) 归一化层
        self.Norm = RMSNorm(dim=tiaoyu_config.embed_dim,     # 输入维度大小, 可以理解成embedding的维度
                            epsilon=tiaoyu_config.Norm_epsilon)  # 归一化层中的epsilon值
        # (6) 输出层
        if USE_FAIRSCALE:
            self.Linear = ColumnParallelLinear(input_size=tiaoyu_config.embed_dim,    # 输入特征的维度
                                               output_size=tiaoyu_config.vocab_size,  # 输出特征的维度
                                               bias=False)                            # 是否使用偏置项
        else:
            self.Linear = nn.Linear(in_features=tiaoyu_config.embed_dim,    # 输入特征的维度
                                    out_features=tiaoyu_config.vocab_size,  # 输出特征的维度
                                    bias=False)                             # 是否使用偏置项
        # 如果设置了共享嵌入矩阵和输出层的权重, 则将输出层权重与嵌入层权重绑定(训练过程中会同步更新)
        if self.tiaoyu_config.tied_weights: 
            self.Embedding.weight = self.Linear.weight
        
        # 3. 初始化输出对象(用于存储模型前向传播的输出)
        self.Output = TiaoyuCausalLMOutputWithPast()
    
    # 前向传播函数
    def forward(self,
                token_id: Optional[torch.Tensor] = None,
                start_position: int = 0,
                use_kv_cache: bool = False,
                cache_k: Optional[List[torch.Tensor]] = None,
                cache_v: Optional[List[torch.Tensor]] = None,
                **args):
        """
        前向传播
        Args:
            token_id (torch.Tensor, optional): 输入的token ID, 默认为None.
            start_position (int): 获取旋转位置编码的起始位置, 默认为0.
            use_kv_cache (bool): 是否使用键值缓存. 默认为False. 在生成过程中, 可以通过设置use_kv_cache=True来启用缓存.
            cache_k (List[torch.Tensor], optional): 之前生成的 key 张量缓存列表. 通过复用这些缓存值, 可以避免重复计算, 从而加速生成过程.
            cache_v (List[torch.Tensor], optional): 之前生成的 value 张量缓存列表. 通过复用这些缓存值, 可以避免重复计算, 从而加速生成过程.
            **args: 其他可选参数.
        Returns:
            self.Output: 包含logits, aux_loss, cache_k 和 cache_v 的TiaoyuCausalLMOutputWithPast对象.
        """
        # 初始化 key 和 value 缓存列表，每个列表的长度为模型层数
        if cache_k is None:
            cache_k = [None] * self.tiaoyu_config.layer_num
        if cache_v is None:
            cache_v = [None] * self.tiaoyu_config.layer_num

        # 1. 将输入的token_id转换为嵌入向量，并应用dropout(token_id->嵌入层->Dropout-h).
        h = self.Dropout(self.Embedding(token_id))

        # 2. 从预计算的旋转位置编码复数形式笛卡尔坐标中截取合适的长度(起始位置为start_position，长度为token_id的长度)
        freqs_cis = self.freqs_cis[start_position:start_position + token_id.size(1)]

        # 3. 通过各解码器层
        for layer, Block in enumerate(self.Blocks): # 遍历解码器的每一层
            # 将 h、位置编码 freqs_cis、是否使用键值缓存 use_kv_cache、key 张量缓存 cache_k、value 张量缓存 cache_v 传递给当前层，
            # 返回更新后的隐藏状态 h 和新的键、值 layer_cache_k 和 layer_cache_v
            h, layer_cache_k, layer_cache_v = Block(
                x=h,                       # 当前的隐藏状态
                freqs_cis=freqs_cis,       # 旋转位置编码复数形式笛卡尔坐标
                use_kv_cache=use_kv_cache, # 是否使用键值缓存
                cache_k=cache_k[layer],    # 之前生成的 key 张量缓存
                cache_v=cache_v[layer]     # 之前生成的 value 张量缓存
            )
            # 更新 key 和 value 缓存列表
            cache_k[layer] = layer_cache_k
            cache_v[layer] = layer_cache_v
        
        # 4. 归一化层
        h = self.Norm(h)

        # 5. 输出层(线性层)
        logits = self.Linear(h)

        # 一点释疑: 
        # 在Transformer的Decoder框架图中, 可以看到Linear层之后还有一个Softmax层.
        # 但是, 在绝大多数公开的大语言模型代码中并没有显式地使用Softmax层. 这是因为:
        # a. 语言模型最常用的损失函数是交叉熵损失函数(CorssEntropy Loss). 在Pytorch
        #    和TensorFlow中, 交叉熵损失函数内部已经包含了Softmax操作. 它们只需要接受
        #    线性层输出的logits作为输入即可.
        # b. 在推理时, 不需要对输出进行Softmax操作. 因为此时只需要对logits进行argmax
        #    操作即可. 而不需要对其进行复杂的Softmax操作.
        # c. 在交叉熵损失函数对Softmax的实现使用了数值优化技术，避免直接计算指数值, 从
        #    而避免了不必要的计算开销, 并减少数值不稳定的风险.

        # 5. 计算MoE前馈神经网络的总辅助损失
        aux_loss = sum(Block.Moe_feed_forward.aux_loss for Block in self.Blocks) # 各层的MoE前馈神经网络辅助损失之和

        # 6. 准备输出
        self.Output.__setitem__('logits', logits)      # 将 logits 添加到 self.Output 中
        self.Output.__setitem__('aux_loss', aux_loss)  # 将 aux_loss 添加到 self.Output 中
        self.Output.__setitem__('cache_k', cache_k)    # 将 cache_k 添加到 self.Output 中
        self.Output.__setitem__('cache_v', cache_v)    # 将 cache_v 添加到 self.Output 中
        return self.Output

    # 生成函数
    @torch.inference_mode() # 该装时期确保在调用相应的函数时，所有的张量操作都被置于推理模式(inference mode)
    def generate(self, 
                 token_id: Optional[torch.Tensor],  # 输入的token ID
                 stream: bool = False,              # 是否使用流式生成, 默认为False
                 repetition_penalty: float = 1.,    # 重复惩罚系数, 用于控制生成的重复度(<1表示鼓励重复, >1表示惩罚重复)
                 temperature: float = 1.,           # 温度参数, 用于控制生成的随机性
                 top_p: float = 0.90,               # 累积概率阈值, 用于控制生成的多样性
                 use_kv_cache=True,                 # 是否使用缓存, 默认为True
                 **args):
        # 1. 检查输入
        # (1) 检查输入的token ID是否为空
        if token_id is None:
            raise ValueError("输入的token ID不能为空!")
        # (2) 检查输入的token ID的长度是否超过了模型的最大序列长度(如果超过了，将其裁剪至最大序列长度-256)
        if token_id.size(1) > self.tiaoyu_config.max_seq_len:
            token_id = token_id[:, -(self.tiaoyu_config.max_seq_len - 256):]
        # (3) temperature取值范围检查
        if temperature < 0:
            raise ValueError("温度参数temperature不能小于0!")
        # (4) top_p取值范围检查
        if top_p <= 0:
            raise ValueError("累积概率阈值top_p必须大于0!")
        elif top_p > 1:
            print(f"top_p={top_p}>1, 强制修改为1!")
            top_p = 1.0
        # (5) 检查重复惩罚系数的取值范围
        if repetition_penalty < 0:
            raise ValueError("重复惩罚系数repetition_penalty不能小于0!")

        # 2. 执行流式生成
        if stream:
            # 直接调用_stream函数执行流式生成
            return self._stream(token_id=token_id, 
                                use_kv_cache=use_kv_cache, 
                                repetition_penalty=repetition_penalty, 
                                temperature=temperature, 
                                top_p=top_p,
                                **args)

        # 3. 执行非流式生成(本质上是循环调用流式输出函数)
        generated_token_id = []   # 创建空的列表, 用于存储生成的token ID
        for i in range(token_id.size(0)):   # 遍历输入的token ID序列(每个批次)
            # a. 获取当前批次的非padding部分, 并保持维度不变
            non_pad_token_id = token_id[i][token_id[i] != self.tiaoyu_config.pad_token_id].unsqueeze(0)
            # b. 调用_stream函数执行流式生成, 获取输出结果(一个生成器对象, 当使用 for 循环迭代这个生成器时, 它会依次返回值)
            stream_yield = self._stream(token_id=non_pad_token_id, 
                                        use_kv_cache=use_kv_cache, 
                                        repetition_penalty=repetition_penalty, 
                                        temperature=temperature,
                                        top_p=top_p,
                                        **args)
            # c. 遍历生成器对象, 获取 token ID 组成的列表
            token_id_yield = [token_id[:, -1:] for tokens in stream_yield]
            # d. 将的列表中的 token ID拼接在一起
            token_id_gen = torch.cat(token_id_list, dim=-1) if token_id_yield else non_pad_token_id
            # e. 将生成的token ID与输入的token ID拼接在一起, 形成完整的序列
            full_sequence = torch.cat([non_pad_token_id, token_id_gen], dim=-1)
            # f. 将当前批次的token ID添加到生成的token ID列表中
            generated_token_id.append(full_sequence)
        # 4. 将生成的token ID列表转换为张量, 并填充到最大长度
        # a. 获取生成的token ID列表的最大长度(多个批次中最大的长度)
        max_length = max(seq.size(1) for seq in generated_token_id)
        # b. 将生成的token ID列表填充到最大长度
        generated_token_id = [
            torch.cat( # torch.cat用于将给定的张量序列沿着指定的维度拼接起来
                [seq, 
                 torch.full(size=(1, max_length - seq.size(1)), 
                            fill_value=self.tiaoyu_config.pad_token_id, 
                            dtype=seq.dtype, 
                            device=seq.device)],
                dim=-1)
            for seq in generated_token_id]
        # 5. 将生成的token ID列表转换为张量, 并返回
        return torch.cat(generated_token_id, dim=0)
    
    # 流式生成函数
    def _stream(self, 
                token_id,            # 输入的token ID
                use_kv_cache,        # 是否使用缓存
                repetition_penalty,  # 重复惩罚系数, 前面出现过的 token, 对应的logits被缩放, 用于控制生成的重复度
                temperature,         # 温度参数, 全体logits被处理的更加平滑或尖锐, 用于控制生成的随机性
                top_p,               # 累积概率阈值, 系统会选择使得累积概率达到这个阈值的最小数量的词作为候选集, 作用与temperature相似
                **args):
       
        # 1. 初始化输出的起始位置
        output_start_position = token_id.shape[1]
        
        # 2. 初始化是否是第1轮生成(第1轮生成是没有 cache_k 和 cache_v 缓存的, 因此需要单独处理)
        is_first_round = True
         
        # 3. 执行生成
        while token_id.shape[1] < self.tiaoyu_config.max_seq_len - 1: # 当token_id的长度小于最大序列长度时, 继续生成

            # (1) 将token_id作为输入, 获取模型的输出
            if is_first_round or not use_kv_cache: # 如果是第1轮生成或者不使用缓存
                output = self(token_id=token_id, 
                              start_position=0,
                              use_kv_cache=use_kv_cache,
                              **args)
                first_round = False
            else:                            # 否则, 使用缓存
                output = self(token_id=token_id[:, -1:], 
                              start_position=token_id.shape[1] - 1,
                              use_kv_cache=use_kv_cache, 
                              cache_k=cache_k,
                              cache_v=cache_v, 
                              **args)
            # (2) 从模型输出output中获取logits和cache_k、cache_v
            logits = output.logits[:, -1, :]
            cache_k = output.cache_k
            cache_v = output.cache_v
            
            # (3) 根据重复惩罚系数对logits进行处理
            # 首先, 将token_id转换为列表, 然后使用集合去重, 得到token_id中出现过的token ID列表
            # 接着, 将logits中对应token ID的logits值除以重复惩罚系数, 这样就可以避免重复生成相同的token ID
            # 最后, 将处理后的logits重新赋值给logits变量, 以便后续使用
            # 简言之, 就是对对先前出现过的token ID的logits值进行惩罚, 降低生成的token ID的概率
            # 所以, repetition_penalty > 1, 表示惩罚因子, 值越大, 惩罚越严重, token ID重复的可能性越低;
            #      repetition_penalty < 1, 表示鼓励因子, 值越小, 鼓励越严重, token ID重复的可能性越高.
            logits[:, list(set(token_id.tolist()[0]))] /= repetition_penalty

            # (4) 根据温度参数对logits进行处理
            # 本质上是对logits进行缩放
            # 当temperature > 1, logits 的绝对值被缩小，导致指数运算后的值更加接近, 生成的token ID的随机性越高;
            # 当temperature < 1, logits 的绝对值被放大，导致指数运算后的值差距更大, 生成的token ID的随机性越低.
            logits /= (temperature + 1e-9)

            # (5) 根据累积概率阈值对logits进行处理
            # 本质上是计算累积概率, 然后根据累积概率阈值, 将logits中累积概率超过阈值的logits值设置为-inf, 
            # 这样就确定了一个包含多个词的小集合, 下一词的输出在这个小集合内进行随机采样. 
            # 这确保了选择的词既有一定的多样性, 又不会完全偏离模型认为最有可能的结果.
            if top_p < 1.0: # 如果top_p < 1.0, 即需要进行top_p处理, 否则不处理
                # a. 将 logits 转换为概率分布
                probs = F.softmax(logits, dim=-1)
                # b. 按概率从高到低排序, 得到排序后的概率和对应的索引
                sorted_probs, sorted_indices = torch.sort(probs, descending=True, dim=-1)
                # c. 计算累积概率
                cumulative_probs = torch.cumsum(sorted_probs, dim=-1) 
                # d. 提取累积概率大于 top_p 的索引
                remove_indices = sorted_indices[cumulative_probs > top_p][1:]
                # e. 将这些索引对应的logits设置为-inf
                logits[0, remove_indices] = -float('Inf')
                # 这里不直接对概率probs进行处理(比如直接将累积概率超过top_p的概率设为0)的原因在于:
                # 直接处理概率, 会使得概率分布的总和不为1, 可能会导致生成的token ID的概率分布不符合预期.
                # 而通过对logits进行处理, 可以保持生成的token ID的概率分布的总和为1.

            # (6) 基于概率采样一个token ID, 作为新生成的token ID
            generated_token_id = torch.multinomial(F.softmax(logits, dim=-1), num_samples=1)
            
            # (7) 将新生成的 generated_token_id 添加到token_id中, 并更新输出起始位置
            token_id = torch.cat((token_id, generated_token_id), dim=1)
            yield token_id[:, output_start_position:]

            # (8) 如果生成的token ID是结束符号 或 达到最大序列长度, 则停止生成
            if (generated_token_id.item() == self.tiaoyu_config.eos_token_id) | (token_id.shape[1] >= self.tiaoyu_config.max_seq_len):
                break


if __name__ == '__main__':
    tiaoyu_config = TiaoyuConfig()
    tiaoyu_model = TIAOYU(tiaoyu_config)
    print('*' * 50, 'tiaoyu_model', '*' * 50)
    print(tiaoyu_model)
    print(f"模型的可训练参数量: {sum(p.numel() for p in tiaoyu_model.parameters() if p.requires_grad)}")

    token_id = torch.randint(0, 8192, (2, 512))
    print('*' * 50, 'input', '*' * 50)
    print(f'token_id: \n{token_id}\n')
    print(f'token_id shape: \n{token_id.shape}\n')
    output = tiaoyu_model(token_id)
    print('*' * 50, f'output of training: {tiaoyu_model.training}', '*' * 50)
    print(f'output: \n{output}\n')
    print(f'output.logits shape: \n{output.logits.shape}\n')

    tiaoyu_model.eval()
    output = tiaoyu_model(token_id)
    print('*' * 50, f'output of training: {tiaoyu_model.training}', '*' * 50)
    print(f'output: \n{output}\n')
    print(f'output shape: \n{output.logits.shape}\n')