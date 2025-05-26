import json
from torch.utils.data import Dataset
import torch
import os # 导入os模块，用于处理文件和目录

os.environ["TOKENIZERS_PARALLELISM"] = "false"


# 定义了一个名为SFTDataset的类，它继承自torch.utils.data模块的Dataset类
# 用于加载和处理预训练模型所需的数据集
"""
SFT数据集格式如下：
{"conversations": [{"role": "user", "content": "你好，我有一个问题想问。"}, 
                   {"role": "assistant", "content": "你好！当然可以，有什么问题你可以尽管问，我会尽力帮助你。"}]}
"""

class SFTDataset(Dataset):
    def __init__(self, data_path, tokenizer, max_seq_len):
        super().__init__()               # 调用父类的构造函数
        self.tokenizer = tokenizer       # 将传入的分词器赋值给实例变量 tokenizer
        self.max_seq_len = max_seq_len   # 将传入的最大长度赋值给实例变量 max_seq_len 
        self.data_list = self.load_data(data_path)  # 调用 load_data 方法加载数据集，并将读取的数据列表赋值给 data_list
        self.bos_id = tokenizer('<s>assistant', add_special_tokens=False).input_ids # 获取<s>assistant的id
        self.eos_id = tokenizer('</s>', add_special_tokens=False).input_ids         # 获取</s>的id

    def __len__(self):
        """
        返回样本数量。当使用内置函数len()来获取这类对象的长度时，Python实际上会调用该对象的__len__方法。
        如果不定义该方法，在执行后续的for循环时会抛出TypeError异常，提示对象不可迭代。
        Args:
            无
        Returns:
            int: 数据数量
        """
        return len(self.data_list)

    def load_data(self, data_path):
        """
        加载数据文件并返回数据列表。
        Args:
            path (str): 数据文件的路径，是单个文件路径。
        Returns:
            list: 包含样本数据的列表。每个样本是一个字典，包含对话信息。
        """
        data_list = []
        with open(data_path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                data = json.loads(line.strip())
                data_list.append(data)
        return data_list

    def _create_chat_prompt(self, conversations):
        """
        构建符合ChatML格式的对话。

        输出数据格式如下：
            <s>system
            你是一个AI助手。</s>
            <s>user
            你好，我是张华，今年15岁，是北京四中的学生。我喜欢编程和篮球，我梦想成为一名软件工程师。很高兴认识你！</s>
            <s>assistant
            你好，张华！很高兴认识你。15岁就有明确的兴趣和梦想非常棒！编程和篮球都是很好的爱好，它们不仅能培养你的逻辑思维和团队合作能力，还能让你在学习之余保持活力。如果你需要任何关于编程学习的建议或资源，或者只是想聊聊篮球，我都非常乐意帮助你。加油，希望你早日实现成为软件工程师的梦想！</s>
        """
        messages = []
        # 遍历对话中的每个轮次，将其中的user和assistant的content拼接起来，然后添加到messages列表中
        for i, turn in enumerate(conversations):
            if turn['role'] == 'user':
                messages.append({"role": 'user', "content": turn['content']})
            else: 
                messages.append({"role": 'assistant', "content": turn['content']})
        
        # 调用tokenizer的apply_chat_template方法，将messages列表中的对话转换为符合ChatML格式的字符串。
        return self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False
        )

    def _generate_loss_mask(self, input_ids):
        """
        生成动态损失掩码。
        对于每个对话，只计算assistant的输出部分的损失。
        """
        loss_mask = [0] * len(input_ids) # 初始化损失掩码为全0，长度与input_ids相同
        i = 0                            # 初始化索引为0
        while i < len(input_ids):        # 循环遍历input_ids中的每个元素
            if input_ids[i:i + len(self.bos_id)] == self.bos_id: # 如果当前位置当前位置的id是bos_id
                start = i + len(self.bos_id)       # 计算assistant输出的起始位置
                end = start                        # 初始化assistant输出的结束位置
                while end < len(input_ids):        # 如果当前结束位置小于input_ids的长度，则循环更新直到得到assistant输出的真实结束位置
                    if input_ids[end:end + len(self.eos_id)] == self.eos_id: # 如果当前结束位置的id是eos_id
                        break
                    end += 1                       # 更新assistant输出的结束位置
                # 对于assistant输出的每个位置，将其对应的损失掩码置为1
                for j in range(start + 1, min(end + len(self.eos_id) + 1, self.max_seq_len)): 
                    loss_mask[j] = 1
                # 更新索引为assistant输出的结束位置加上eos_id的长度，如果超过了input_ids的长度，则更新为input_ids的长度
                i = end + len(self.eos_id) if end < len(input_ids) else len(input_ids)
            else:
                i += 1
        return loss_mask

    def __getitem__(self, index):
        """
        该函数使类的实例能够像序列（如列表或元组）一样通过索引访问元素。
        Args:
            index (int): 要获取的样本的索引。
        Returns:
            tuple: 包含处理后的输入数据 (X, Y, loss_mask) 的元组。
                - X (torch.Tensor): 输入数据的特征向量。
                - Y (torch.Tensor): 目标数据的特征向量。
                - loss_mask (torch.Tensor): 损失掩码，用于在计算损失时忽略填充的部分。
        """
        # (1) 从数据列表中获取指定索引的数据
        sample = self.data_list[index]
        # print(f"sample: {sample}")
        # (2) 创建对话prompt
        prompt = self._create_chat_prompt(sample['conversations'])
        # print(f"prompt: {prompt}")
        # (3) 使用分词器对prompt进行分词，并截断至最大长度
        input_ids = self.tokenizer(prompt).input_ids[:self.max_seq_len]
        # (4) 对input_ids进行填充，使其长度达到最大长度
        input_ids += [self.tokenizer.pad_token_id] * (self.max_seq_len - len(input_ids))
        # (5) 生成动态损失掩码
        loss_mask = self._generate_loss_mask(input_ids)

        # 构建训练数据
        X = torch.tensor(input_ids[:-1], dtype=torch.long)
        Y = torch.tensor(input_ids[1:], dtype=torch.long)
        loss_mask = torch.tensor(loss_mask[1:], dtype=torch.long)  # 对齐预测位置

        return X, Y, loss_mask


if __name__ == "__main__":
    from transformers import AutoTokenizer
    from torch.utils.data import DataLoader

    # 加载分词器
    BPE_tokenizer = AutoTokenizer.from_pretrained("./model/BPE_tokenizer")
    # 实例化PretrainDataset类
    pretrain_dataset = SFTDataset(data_path='./data/sft_data/train_data/sft.jsonl', tokenizer=BPE_tokenizer, max_seq_len=128)
    # 创建数据加载器
    train_loader = DataLoader(
        dataset=pretrain_dataset,
        batch_size=1,
    )
    print(len(train_loader))

    # 遍历数据加载器
    for step, (X, Y, loss_mask) in enumerate(train_loader):
        print('X: ', X)
        print('Y: ', Y)
        print('loss_mask: ', loss_mask)
        print(f'loss_mask_sum: {loss_mask.sum()}')
        if step == 1:
            break