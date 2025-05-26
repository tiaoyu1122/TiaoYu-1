import os
import json
from datasets import load_dataset
from tokenizers import (
    Tokenizer,
    models,
    pre_tokenizers,
    trainers,
    decoders
)
import random
import logging

random.seed(42)   # 设置随机种子，以便于复现结果
logging.basicConfig(level=logging.INFO) # 设置日志级别为INFO，以便于查看日志信息


"""
训练并保存一个分词器(tokenizer)。分词器可以将文本分割成词元，并将这些词元映射到一个整数索引。
即，一段文本可以通过分词器转换为一组数字编码，从而可以进行计算机处理。
本段代码即构建了一个BPE_tokenizer_train函数，用于训练并保存一个基于字节对编码(BPE)的分词器。
"""

def BPE_tokenizer_train(data_path: str = './data/pretrain_data/pretrain_data.json',
                        percent: float = 1.0,
                        vocab_size: int = 32768,
                        output_dir: str = "./model/BPE_tokenizer"):
    """
    训练并保存一个基于字节对编码(BPE)的分词器。
    Args:
        data_path (str, optional): 训练数据JSONL文件路径，默认值为 './data/pretrain_data/pretrain_data.json'，即采用预训练数据。
        percent (float, optional): 训练数据采样的比例，取值范围(0,1]，默认值为 1.0。
        vocab_size (int, optional): 词汇表大小，默认值为 32768。
        output_dir (str, optional): 分词器保存目录，默认值为 './model/BPE_tokenizer'。
    Returns:
        None
    """
    # (1) 检验输入参数取值是否合理 -----------------------------------------------------------------------------------------
    logging.info('(1) 检验输入参数取值是否合理...')
    # 检验percent是否在(0,1]范围内
    if not (0 < percent <= 1): 
        logging.error("percent 必须在 (0, 1] 范围内")
        return
    # 检验vocab_size是否大于0
    if vocab_size <= 0:
        logging.error("vocab_size 必须大于 0")
        return

    # (2) 初始化BPE分词模型 ----------------------------------------------------------------------------------------------
    logging.info('(2) 初始化BPE分词模型...')
    BPE_tokenizer = Tokenizer(models.BPE())
    # 设置预分词器为ByteLevel
    BPE_tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)

    # (3) 定义训练器并设置训练参数------------------------------------------------------------------------------------------
    logging.info('(3) 定义训练器并设置训练参数...')
    added_tokens = ["<unk>", "<s>", "</s>"]        # 预定义的标记列表, 包括未知token(<unk>)、开始token(<s>)、结束token(</s>)
    trainer = trainers.BpeTrainer(                 # 使用tokenizers库来创建一个BpeTrainer对象，用于训练一个基于字节对编码(BPE)的分词器
        vocab_size = vocab_size,                   # 指定目标词汇表的大小，即希望生成的分词器能够包含的不同词元的数量
        special_tokens = added_tokens,             # 指定预定义的标记, 确保这些标记被包含在最终的词汇表中
        show_progress = True,                      # 控制是否在训练过程中显示进度信息
        initial_alphabet = pre_tokenizers.ByteLevel.alphabet(), # 指定初始字母表，用于字节级别的预处理. pre_tokenizers.ByteLevel.alphabet()提供了一个基于字节的初始字母表，这意味着分词器会首先基于字节对文本进行分割，然后再应用BPE算法。这对于处理包含非拉丁字符的文本特别有用，因为它允许算法在字符级别以下（即字节级别）进行操作.
        min_frequency=1,                           # 指定词频阈值，用于确定哪些词元应该被包含在词汇表中
        num_threads=4                              # 指定用于训练的线程数
    ) # 目前tokenizers库并不直接支持GPU加速，它主要依赖多线程CPU并行化来加速训练过程

    # (4) 读取训练使用的文本数据(得到一个生成器)------------------------------------------------------------------------------
    logging.info('(4) 读取训练使用的文本数据(得到一个生成器)...')
    # 定义一个text_pull函数，读取data_path中的JSONL文件并按percent比例随机提取文本数据，返回一个生成器
    def text_pull(file_path, percent):
        try:
            with open(file_path, 'r', encoding='utf-8') as f: # 打开文件
                lines = f.readlines() # 读取所有行
                sample_size = min(int(len(lines) * percent), len(lines)) # 计算按percent采样后的样本大小(行数)
                sampled_lines = random.sample(lines, sample_size)        # 从lines中随机抽取sample_size个样本
                for line in sampled_lines:                               # 遍历采样后的样本
                    try:
                        data = json.loads(line)                          # 将JSON格式的字符串转换为Python字典
                        yield data['text']                               # 返回一个生成器，每次迭代返回一个文本数据
                    except json.JSONDecodeError as e:
                        logging.warning(f'第 {str(line)} 行的JSON格式错误！')
                        continue
        except FileNotFoundError as e:
            logging.error(f'文件未找到: {file_path}')
    # 调用text_pull函数，读取data_path中的JSONL文件并按percent比例随机提取文本数据，返回一个生成器
    texts = text_pull(data_path, percent) 
    
    # (5) 训练tokenizer并设置解码器----------------------------------------------------------------------------------------
    logging.info('(5) 训练tokenizer...')
    BPE_tokenizer.train_from_iterator(texts, trainer=trainer) # 根据提供的文本数据texts来训练一个分词器（tokenizer），通过trainer参数来指定训练过程中使用的配置或参数
    BPE_tokenizer.decoder = decoders.ByteLevel() # 设置分词器的解码器(用于将模型输出的数值形式转换回原始文本或接近原始文本的格式)    

    # (6) 手动创建tokenizer配置文件 ---------------------------------------------------------------------------------------
    # 用于设置分词器（tokenizer）的行为，特别是在处理特殊标记（如开始标记、结束标记、未知标记等）和文本格式等
    logging.info('(6) 手动创建tokenizer配置文件...')
    config = {
        "add_bos_token": False,      # 是否在序列的开始添加开始标记（Begin Of Sentence，BOS）
        "add_eos_token": False,      # 是否在序列的末尾添加结束标记（End Of Sentence，EOS）
        "add_prefix_space": False,   # 在分词时是否在标记前添加空格
        "added_tokens_decoder": {    # 定义了额外添加的标记及其属性
            str(i): {
                "content": token,
                "lstrip": False,
                "normalized": False,
                "rstrip": False,
                "single_word": False,
                "special": True
            } for i, token in enumerate(added_tokens)
        },
        "additional_special_tokens": [], # 额外的特殊标记列表
        "bos_token": "<s>",              # 开始标记
        "clean_up_tokenization_spaces": False, # 是否在分词后清理多余的空格
        "eos_token": "</s>",             # 结束标记
        "legacy": True,                  # 是否使用旧的配置文件格式
        "model_max_length": 65536,       # 模型能处理的最大序列长度
        "pad_token": "<unk>",            # 填充标记
        "sp_model_kwargs": {},           # 传递给子词模型（如SentencePiece）的额外参数
        "spaces_between_special_tokens": False, # 在特殊标记之间是否添加空格
        "tokenizer_class": "PreTrainedTokenizerFast", # 分词器类型的名称
        "unk_token": "<unk>",            # 未知标记
        "chat_template": "{% if messages[0]['role'] == 'system' %}{% set system_message = messages[0]['content'] %}{{ '<s>system\\n' + system_message + '</s>\\n' }}{% else %}{{ '<s>system\\n你是一个AI助手。</s>\\n' }}{% endif %}{% for message in messages %}{% set content = message['content'] %}{% if message['role'] == 'user' %}{{ '<s>user\\n' + content + '</s>\\n<s>assistant\\n' }}{% elif message['role'] == 'assistant' %}{{ content + '</s>' + '\\n' }}{% endif %}{% endfor %}" # 格式化聊天消息的模板字符串
    }
    
    # (7) 保存tokenizer -------------------------------------------------------------------------------------------------
    logging.info('(7) 保存tokenizer...')
    os.makedirs(output_dir, exist_ok=True) # 创建目录，如果目录不存在则创建目录
    BPE_tokenizer.save(os.path.join(output_dir, "tokenizer.json")) # 保存分词器tokenizer.json
    BPE_tokenizer.model.save(output_dir)   # 保存分词器的完整模型(merges.txt, vocab.json)
    # 保存配置文件tokenizer_config.json
    with open(os.path.join(output_dir, "tokenizer_config.json"), "w", encoding="utf-8") as config_file:
        json.dump(config, config_file, ensure_ascii=False, indent=4)

    logging.info("tokenizer训练并保存完成!!!")


if __name__ == '__main__':
    
    # 如果已经有训练好的tokenizer，可以直接加载使用，不需要再次训练，将下面代码注释掉即可

    # 训练并保存一个基于字节对编码(BPE)的分词器
    print('#' * 50, '训练开始', '#' * 50)
    BPE_tokenizer_train(data_path = './data/pretrain_data/train_data/pretrain_hq.jsonl',
                        percent = 0.6,
                        vocab_size = 8192,
                        output_dir = "./model/BPE_tokenizer")
    
    # 验证BPE_tokenizer是否正确
    print('#' * 50, '验证开始', '#' * 50)
    from transformers import AutoTokenizer
    # 从"./model/BPE_tokenizer"加载预训练的tokenizer
    BPE_tokenizer = AutoTokenizer.from_pretrained("./model/BPE_tokenizer")
    # 设置验证对话消息列表
    messages = [{"role": "system", "content": "你是一个AI助手。"},
                {"role": "user", "content": '卡尔·马克思出生于哪个国家？'},
                {"role": "assistant", "content": '德国。'}]
    print(f'(a) 原始输入的对话消息列表:\n{messages}')
    # 使用BPE_tokenizer的apply_chat_template方法将消息列表转换为一个格式化的字符串(config中定义的chat_template)，这里tokenize=False表示不将文本分割成tokens
    formatted_message = BPE_tokenizer.apply_chat_template(messages,  tokenize=False)
    print(f'(b) 格式化后的对话消息列表:\n{formatted_message}')
    # 使用BPE_tokenizer将formatted_message分割成tokens，并将这些tokens映射到对应的ID上
    encode_result = BPE_tokenizer(formatted_message)
    print(f'(c) 分词器encode结果:\n{encode_result}')
    # 使用BPE_tokenizer的decode方法(在模型训练的第(5)步中曾设置)将词元编码encode_result['input_ids']转换回文本
    decode_result = BPE_tokenizer.decode(encode_result['input_ids'], skip_special_tokens=False)
    print(f'(d) decode结果:\n{decode_result}')
    print(f'(e) decode和格式化文本是否一致: {decode_result == formatted_message}')

