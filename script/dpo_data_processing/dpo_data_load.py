import json
from torch.utils.data import Dataset
import torch
import os # 导入os模块，用于处理文件和目录

os.environ["TOKENIZERS_PARALLELISM"] = "false"


class DPODataset(Dataset):
    def __init__(self, data_path, tokenizer, max_seq_len=4096):
        super().__init__()             # 调用父类的构造函数
        self.tokenizer = tokenizer     # 将传入的分词器赋值给实例变量 tokenizer
        self.max_seq_len = max_seq_len # 将传入的最大长度赋值给实例变量 max_seq_len 
        self.padding = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0 # 获取填充符号的id
        self.bos_id = tokenizer('<s>assistant', add_special_tokens=False).input_ids        # 获取<s>assistant的id
        self.eos_id = tokenizer('</s>', add_special_tokens=False).input_ids                # 获取</s>的id
        # 打开文件，并将每一行转换为json对象，然后将其添加到self.data列表中
        with open(data_path, 'r', encoding='utf-8') as f:
            self.data = []
            for line in f:
                line = line.strip()
                obj = json.loads(line)
                self.data.append(obj)

    def __len__(self):
        """
        返回样本数量。当使用内置函数len()来获取这类对象的长度时，Python实际上会调用该对象的__len__方法。
        如果不定义该方法，在执行后续的for循环时会抛出TypeError异常，提示对象不可迭代。
        Args:
            无
        Returns:
            int: 数据数量
        """
        return len(self.data)

    def _generate_loss_mask(self, input_ids):
        """
        生成动态损失掩码。
        对于每个对话，只计算assistant的输出部分的损失。
        """
        loss_mask = [0] * len(input_ids)  # 初始化损失掩码为全0，长度与input_ids相同
        i = 0                             # 初始化索引为0
        while i < len(input_ids):         # 循环遍历input_ids中的每个元素
            if input_ids[i:i + len(self.bos_id)] == self.bos_id: # 如果当前位置当前位置的id是bos_id
                start = i + len(self.bos_id)       # 计算assistant输出的起始位置
                end = start                        # 初始化assistant输出的结束位置
                while end < len(input_ids):        # 如果当前位置小于input_ids的长度，则循环更新直到得到assistant输出的结束位置
                    if input_ids[end:end + len(self.eos_id)] == self.eos_id: # 如果当前结束位置的id是eos_id
                        break
                    end += 1  # 更新assistant输出的结束位置
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
            字典, 包括:
             - x_chosen (torch.Tensor): 选中的输入序列特征向量
             - y_chosen (torch.Tensor): 选中的输出序列特征向量
             - mask_chosen (torch.Tensor): 选中的损失掩码，用于在计算损失时忽略填充的部分
             - x_rejected (torch.Tensor): 拒绝的输入序列特征向量
             - y_rejected (torch.Tensor): 拒绝的输出序列特征向量
             - mask_rejected (torch.Tensor): 拒绝的损失掩码，用于在计算损失时忽略填充的部分
        }
        """

        item = self.data[index] # 获取数据集中的第index个元素
        """
        数据形如:
        {"chosen": [{"content": "A curve in the plane is defined parametrically by the equations $x=2\\cos(3t)$ and $y=3\\sin(2t)$. Find $\\frac{dy}{dx}$.", "role": "user"}, 
                    {"content": "To find $\\frac{dy}{dx}$ for a curve defined parametrically by $x(t)$ and $y(t)$, we use the formula:\n\n\\[\n\\frac{dy}{dx} = \\frac{\\frac{dy}{dt}}{\\frac{dx}{dt}}\n\\]\n\nFirst, we find $\\frac{dx}{dt}$ and $\\frac{dy}{dt}$ for the given parametric equations:\n\n\\[\nx(t) = 2\\cos(3t) \\quad \\Rightarrow \\quad \\frac{dx}{dt} = -6\\sin(3t)\n\\]\n\n\\[\ny(t) = 3\\sin(2t) \\quad \\Rightarrow \\quad \\frac{dy}{dt} = 6\\cos(2t)\n\\]\n\nNow, we can find $\\frac{dy}{dx}$:\n\n\\[\n\\frac{dy}{dx} = \\frac{\\frac{dy}{dt}}{\\frac{dx}{dt}} = \\frac{6\\cos(2t)}{-6\\sin(3t)} = -\\frac{\\cos(2t)}{\\sin(3t)}\n\\]\n\nThus, the derivative of $y$ with respect to $x$ is:\n\n\\[\n\\boxed{\\frac{dy}{dx} = -\\frac{\\cos(2t)}{\\sin(3t)}}\n\\]", "role": "assistant"}], 
         "rejected": [{"content": "A curve in the plane is defined parametrically by the equations $x=2\\cos(3t)$ and $y=3\\sin(2t)$. Find $\\frac{dy}{dx}$.", "role": "user"}, 
                      {"content": "To calculate the derivative of y with respect to x, we need to apply the chain rule. The chain rule states that if f(x)=g(h(x)), where g' and h' exist, then f'(x)=h'(x)*g'(h(x)). \n\nIn our case, we have y=g(t)=3*sin(2*t) and x=h(t)=2*cos(3*t).\n\nSo, dy/dx=(d/dt)(3*sin(2*t))/(d/dt)(2*cos(3*t))=6*2*cos(3*t)*(-sin(2*t))/(-9*2*sin(3*t)*(-cos(3*t)))=-0.66667*(tan(2*t)/tan(3*t))\n\nPlease note that this answer may vary slightly depending on the rounding precision used during calculation.", "role": "assistant"}]}
        """
        chosen = item['chosen']     # 获取chosen列表
        # print(f'chosen: {chosen}')
        rejected = item['rejected'] # 获取rejected列表
        # print(f'rejected: {rejected}')
        
        # 调用tokenizer的apply_chat_template方法，将对话转换为符合ChatML格式的字符串。
        chosen_prompt = self.tokenizer.apply_chat_template(chosen, tokenize=False, add_generation_prompt=False)
        # print(f'chosen_prompt: {chosen_prompt}')
        rejected_prompt = self.tokenizer.apply_chat_template(rejected, tokenize=False, add_generation_prompt=False)
        # print(f'rejected_prompt: {rejected_prompt}')
        
        # 使用tokenizer对prompt进行分词，并截断至最大长度。
        chosen_encoding = self.tokenizer(chosen_prompt, truncation=True, max_length=self.max_seq_len, padding='max_length')
        rejected_encoding = self.tokenizer(rejected_prompt, truncation=True, max_length=self.max_seq_len, padding='max_length')
        
        # 提取input_ids
        chosen_input_ids = chosen_encoding['input_ids']
        rejected_input_ids = rejected_encoding['input_ids']
        
        # 生成动态损失掩码
        chosen_loss_mask = self._generate_loss_mask(chosen_input_ids)
        rejected_loss_mask = self._generate_loss_mask(rejected_input_ids)

        # 构建训练数据
        X_chosen = torch.tensor(chosen_input_ids[:-1], dtype=torch.long)
        Y_chosen = torch.tensor(chosen_input_ids[1:], dtype=torch.long)
        mask_chosen = torch.tensor(chosen_loss_mask[1:], dtype=torch.long)
        X_rejected = torch.tensor(rejected_input_ids[:-1], dtype=torch.long)
        Y_rejected = torch.tensor(rejected_input_ids[1:], dtype=torch.long)
        mask_rejected = torch.tensor(rejected_loss_mask[1:], dtype=torch.long)

        return {
            'X_chosen': X_chosen,
            'Y_chosen': Y_chosen,
            'mask_chosen': mask_chosen,
            'X_rejected': X_rejected,
            'Y_rejected': Y_rejected,
            'mask_rejected': mask_rejected
        }


if __name__ == "__main__":
    from transformers import AutoTokenizer
    from torch.utils.data import DataLoader

    # 加载分词器
    BPE_tokenizer = AutoTokenizer.from_pretrained("./model/BPE_tokenizer")
    # 实例化PretrainDataset类
    pretrain_dataset = DPODataset(data_path='./data/rlhf_data/rlhf_dpo.jsonl', tokenizer=BPE_tokenizer, max_seq_len=512)
    # 创建数据加载器
    train_loader = DataLoader(
        dataset=pretrain_dataset,
        batch_size=2,
    )
    print(len(train_loader))

    # 遍历数据加载器
    for step, result in enumerate(train_loader):
        print('X_chosen: ', result['X_chosen'])
        print('Y_chosen: ', result['Y_chosen'])
        print('mask_chosen: ', result['mask_chosen'])
        print(f'mask_chosen_sum: {result["mask_chosen"].sum()}')
        print('X_rejected: ', result['X_rejected'])
        print('Y_rejected: ', result['Y_rejected'])
        print('mask_rejected: ', result['mask_rejected'])
        print(f'mask_rejected_sum: {result["mask_rejected"].sum()}')
        if step == 1:
            break