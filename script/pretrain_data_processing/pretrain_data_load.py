import json
from torch.utils.data import Dataset
import torch
import os  # 导入os模块，用于处理文件和目录
import re
from concurrent.futures import ThreadPoolExecutor
from json import JSONDecoder

# 定义了一个名为PretrainDataset的类，它继承自torch.utils.data模块的Dataset类
# 用于加载和处理预训练模型所需的数据集
class PretrainDataset(Dataset):
    def __init__(self, data_path, tokenizer, max_seq_len):
        super().__init__()   # 调用父类的构造函数
        self.tokenizer = tokenizer    # 将传入的分词器赋值给实例变量 tokenizer
        self.max_seq_len = max_seq_len  # 将传入的最大长度赋值给实例变量 max_seq_len
        self.data_list = self.load_data(data_path)  # 调用 load_data 方法加载数据集，并将读取的数据列表赋值给 data_list

    def load_data(self, data_path):
        """
        加载数据文件并返回数据列表。
        Args:
            path (str): 数据文件的路径，可以是单个文件路径或包含多个文件的目录路径。
        Returns:
            list: 包含样本数据的列表。每个样本是一个字典，包含文本数据和标签信息。
        """
        data_list = []                                           # 初始化一个空列表，用于存储数据
        if os.path.isdir(data_path):                                       # 如果path是目录
            for root, dirs, files in os.walk(data_path):                       # 遍历path目录下的所有文件和子目录
                for file in files:                                        # 遍历所有文件
                    file_path = os.path.join(root, file)                  # 拼接文件路径
                    try:
                        with open(file_path, 'r', encoding='utf-8') as f: # 打开文件并读取内容
                            print(f"正在加载文件：{file_path}")             # 打印当前正在加载的文件路径
                            for line_num, line in enumerate(f, 1):        # 逐行读取文件内容
                                data = json.loads(line.strip())           # 解析JSON对象，并将其存储到data中
                                data_list.append(data)                    # 将数据添加到数据列表中
                    except Exception as e:
                        print(f"加载文件 {file_path} 时出错: {e}")          # 打印错误信息
        elif os.path.isfile(data_path):                                    # 如果path是文件
            try:
                with open(data_path, 'r', encoding='utf-8') as f:              # 打开文件并读取内容
                    print(f"正在加载文件：{data_path}")                          # 打印当前正在加载的文件路径
                    for line_num, line in enumerate(f, 1):                # 逐行读取文件内容
                        data = json.loads(line.strip())                   # 解析JSON对象，并将其存储到data中
                        data_list.append(data)                            # 将数据添加到数据列表中
            except Exception as e:
                print(f"加载文件 {data_path} 时出错: {e}")                       # 打印错误信息
        return data_list

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
        # (1) 从数据列表中获取指定索引的数据，并提取文本
        data = f"{str(self.data_list[index]['text'])}"
        # data = re.sub(r'[\x00-\x1F\x7F-\x9F]', '', data) # 去除特殊字符

        # (2) 使用分词器对文本进行编码
        # 结果是一个字典，包含以下键值对：
        # - input_ids: 输入文本的编码表示，每个元素对应一个词汇的索引。
        # - token_type_ids: 标记类型ID，用于区分不同的句子。
        # - attention_mask: 注意力掩码，用于指示哪些位置是有效的。
        # 形如：{'input_ids': tensor([[1, 168, 234, 115, 1415, 333, 1335, 413, 677, 2239, 269, 2397, 315, 2669, 2, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]), 
        #       'token_type_ids': tensor([[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]), 
        #       'attention_mask': tensor([[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]])}
        encode_result = self.tokenizer(
            data,
            max_length=self.max_seq_len,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        ) 
        
        # (3) 获取编码后的输入ID, 生成损失掩码
        # 获取编码后的输入ID
        # 形如：tensor([[1, 168, 234, 115, 1415, 333, 1335, 413, 677, 2239, 269, 2397, 315, 2669, 2, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]])
        input_ids = encode_result.input_ids.squeeze()
        # 生成损失掩码，用于在计算损失时忽略填充的部分
        # 形如：# tensor([True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, False, False, False, False, False, False, False, False, False, False, False, False, False,  False, False])
        loss_mask = (input_ids != self.tokenizer.pad_token_id) 
        
        # (4) 生成输入数据的特征向量X, 目标数据的特征向量Y, 以及损失掩码loss_mask
        # 获取输入数据的特征向量X，去除最后一个ID（结束标记）
        X = input_ids[:-1].clone().detach().long() # tensor([1, 168, 234, 115, 1415, 333, 1335, 413, 677, 2239, 269, 2397, 315, 2669, 2, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
        # 获取目标数据的特征向量Y，从第二个ID开始（跳过开始标记）
        Y = input_ids[1:].clone().detach().long()  # tensor([168, 234, 115, 1415, 333, 1335, 413, 677, 2239, 269, 2397, 315, 2669, 2, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
        # 损失掩码从第二个位置开始，忽略第一个位置（开始标记）
        loss_mask = loss_mask[1:].clone().detach().long() # tensor([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
        
        # 检查数据是否存在 NaN 或 Inf
        if torch.isnan(X).any() or torch.isinf(X).any() or torch.isnan(Y).any() or torch.isinf(Y).any() or torch.isnan(loss_mask).any() or torch.isinf(loss_mask).any():
            print(f"Data at index {index} contains NaN or Inf. Skipping this data.")
            # 可以选择返回默认值或者重新采样
            # 这里简单返回全零数据
            X = torch.zeros_like(X)
            Y = torch.zeros_like(Y)
            loss_mask = torch.zeros_like(loss_mask)

        return X, Y, loss_mask

if __name__ == "__main__":
    from transformers import AutoTokenizer
    from torch.utils.data import DataLoader

    # 加载分词器
    BPE_tokenizer = AutoTokenizer.from_pretrained("./model/BPE_tokenizer")
    # 实例化PretrainDataset类
    pretrain_dataset = PretrainDataset(data_path='./data/pretrain_data/train_data/pretrain_hq.jsonl', tokenizer=BPE_tokenizer, max_seq_len=256)
    # 创建数据加载器
    train_loader = DataLoader(
        dataset=pretrain_dataset,
        batch_size=2,
    )
    print(len(train_loader))

    # 遍历数据加载器
    for step, (X, Y, loss_mask) in enumerate(train_loader):
        print('X: ', X)
        print('Y: ', Y)
        print('loss_mask: ', loss_mask)
        if step == 1:
            break