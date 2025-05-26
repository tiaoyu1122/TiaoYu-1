from transformers import PretrainedConfig
import json

"""
本代码定义了一个名为ModelConfig的类, 用于配置模型的各种超参数。
"""

class TiaoyuConfig(PretrainedConfig):
    """
    TiaoyuConfig类继承自PretrainedConfig类.
    PretrainedConfig 是所有模型配置类的基类. 它定义了模型的基本配置参数(如超参数、架构选项等),  并为具体的模型配置类提供了统一的接口.
    PretrainedConfig 的主要职责是：
     - 定义和存储模型的配置参数
     - 提供加载和保存配置的功能
     - 为子类提供默认值和验证机制
     - 支持从预训练模型加载配置
    """
    def __init__(self,
                 model_name: str = 'TIAOYU',           # 模型名称
                 # 模型结构基础参数配置
                 max_seq_len: int = 2048,              # 模型默认处理的最大文本块长度
                 vocab_size: int = 8192,               # 词汇表的大小
                 embed_dim: int = 512,                 # 模型的嵌入维度
                 layer_num: int = 8,                  # 解码器(注意力机制和前馈神经网络)层数
                 attention_weight_type: str = 'GQA',   # 注意力权重的计算方式，可选'MHA', 'MQA', 'GQA'
                 head_num: int = 8,                    # 多头自注意力机制中头的个数(即Query的头的个数)
                 query_group_num: int = 2,             # 多头自注意力机制中的分组数量(Query被分为query_group_num个组。今且仅当attention_weight_type='GQA'时生效，且query_group_num必须可以被head_num整除。组内共享Key和Value)，当attention_weight_type='MQA'时默认为head_num，当attention_weight_type='MQA'时默认为1.
                 ffn_dim_multiplier: float = 2.5,      # 前馈神经网络层中隐藏层的维度是嵌入层的多少倍
                 tied_weights: bool = True,            # 是否共享嵌入矩阵和输出层的权重
                 # 专家混合机制参数配置
                 expert_num: int = 4,                  # 总的专家数量
                 expert_use: int = 2,                  # 每次启用的专家数量
                 aux_loss_lambda: float = 0.1,         # 辅助损失的权重，要乘到辅助损失上，用于平衡主损失和辅助损失
                 # 模型函数中的系数配置
                 Norm_epsilon: float = 1e-6,           # 归一化层中的epsilon值
                 RoPE_theta: float = 10000.0,          # 旋转位置编码中使用的theta值
                 Dropout_p: float = 0.0,               # dropout比率
                 # 结束标记token
                 vocab_json_file: str = './model/BPE_tokenizer/vocab.json',  # 词汇表文件路径
                 eos_token='</s>',                     # 结束标记token
                 pad_token='<unk>',                    # 填充标记token
                 **kwargs):

        super().__init__(**kwargs)

        self.model_name = model_name

        # 模型结构基础参数配置
        self.max_seq_len = max_seq_len
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.layer_num = layer_num
        self.attention_weight_type = attention_weight_type
        self.head_num = head_num
        if attention_weight_type == 'GQA':
            if head_num % query_group_num != 0:
                raise ValueError("注意力权重的计算方式为GQA时, head_num 必须可被 query_group_num 整除！")
            self.query_group_num = query_group_num
        elif attention_weight_type == 'MQA':
            self.query_group_num = 1
        else:  # attention_weight_type == 'MHA'
            self.query_group_num = head_num
        self.ffn_dim_multiplier = ffn_dim_multiplier
        self.tied_weights = tied_weights

        # 专家混合机制参数配置
        self.expert_num = expert_num
        self.expert_use = expert_use
        self.aux_loss_lambda = aux_loss_lambda

        # 模型函数中的系数配置
        self.Norm_epsilon = Norm_epsilon
        self.RoPE_theta = RoPE_theta
        self.Dropout_p = Dropout_p

        # 从 vocab.json 文件中读取结束标记和填充的ID
        try:
            with open(vocab_json_file, 'r', encoding='utf-8') as f:
                vocab = json.load(f)
            if eos_token in vocab:
                self.eos_token_id = vocab[eos_token]
            else:
                print(f"结束标记 {eos_token} 未在词汇表中找到。")
                self.eos_token_id = None
            if pad_token in vocab:
                self.pad_token_id = vocab[pad_token]
            else:
                print(f"填充标记 {pad_token} 未在词汇表中找到。")
                self.pad_token_id = None
        except FileNotFoundError:
            print("vocab.json 文件未找到。")
            self.eos_token_id = None
        except json.JSONDecodeError:
            print("无法解析 vocab.json 文件。")
            self.eos_token_id = None


if __name__ == '__main__':
    config = TiaoyuConfig()
    print(config.__dict__)