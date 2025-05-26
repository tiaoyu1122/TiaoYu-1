import os
import json
import re
from tqdm import tqdm
import opencc
import logging
import random

"""
预训练数据处理，主要分为以下步骤(这里步骤介绍很简略，可以直接看代码注释): 
1. 从"./data/gross_data/ABear___Wiki_CN/wiki-cn"中读取"wiki_"开头数据的数据文件，如“wiki_0035”；
2. 这个文件中的数据是以json格式存储的，每一行是一个json对象。需要从中提取content字段的内容，并对文本进行一系列的清洗。
3. 将content的内容处理(合并/切分)成长度不少于一定下限的字符串，并且以"<s>"和"</s>"作为开始和结束标记。
4. 处理之后的文本，随机抽取90%作为训练数据，10%作为验证数据。
5. 按最多每个文件100000行，保存到一个结果json文件中，每个json对象只包含一个text字段，内容为处理后的文本。
6. 训练数据保存到"./data/pretrain_data/train_data/"中，命名为"train_data_*.json"，*按照文件名递增的顺序。
7. 验证数据保存到"./data/pretrain_data/eval_data/"中，命名为"eval_data_*.json"，*按照文件名递增的顺序。
"""

logging.basicConfig(level=logging.INFO) # 设置日志级别为INFO，以便于查看日志信息

# (1) 定义输入和输出文件路径 -----------------------------------------------------------------------------------------------
logging.info('(1) 定义输入和输出文件路径...')
input_dir = "./data/gross_data/ABear___Wiki_CN/wiki-cn"  # 下载的预训练数据文件路径

# (2) 创建输出目录（如果不存在）--------------------------------------------------------------------------------------------
logging.info('(2) 创建输出目录（如果不存在）...')
train_dir = "./data/pretrain_data/train_data/"   # 训练数据输出目录
eval_dir = "./data/pretrain_data/eval_data/"     # 验证数据输出目录
os.makedirs(train_dir, exist_ok=True)            # 创建输出目录
os.makedirs(eval_dir, exist_ok=True)             # 创建输出目录

# (3) 遍历输入目录中的所有文件，将它们中的文本内容读入到all_text中---------------------------------------------------------------
logging.info('(3) 遍历输入目录中的所有文件，将它们中的文本内容读入到all_text中...')
start_tag, end_tag = "<s>", "</s>" # 定义起始和结束标记
all_text = "" # 存储所有文本内容，连成一个大字符串，来自不同行的数据分别以start_tag和end_tag包裹
converter = opencc.OpenCC('t2s') # 创建OpenCC对象，用于繁简转换
for filename in os.listdir(input_dir):  # 遍历input_dir中的所有文件
    if filename.startswith("wiki_"):    # 只处理以"wiki_"开头的文件
        file_path = os.path.join(input_dir, filename) # 获取文件路径
        logging.info(f'  - 读取并处理{filename}的文本数据中...')
        # 打开文件并逐行读取
        with open(file_path, 'r', encoding='utf-8') as file:
            lines = file.readlines() # 读取file_path中的所有行
            for line in lines: # 遍历每一行
                try:
                    # 解析JSON对象
                    data = json.loads(line)
                    # 提取content字段的内容
                    content = data.get('content', '')
                    # 删除所有"#"和"\n"
                    while '#' in content:
                        content = content.replace("#", "")
                    while '\n' in content:
                        content = content.replace("\n", "")
                    # 删除（）和其中的内容，因为其中很多都是名词对应的英文翻译，在中文语境下没有必要
                    content = re.sub(r'（.*?）', '', content)
                    # 删除开头和结尾的空格
                    content = content.strip()
                    # 如果文本中全角标点符号前后有空格，则去掉所有空格
                    content = re.sub(r'([\u3000-\u303F\uFF00-\uFFEF])\s+', r'\1', content)
                    content = re.sub(r'\s+([\u3000-\u303F\uFF00-\uFFEF])', r'\1', content)
                    # 如果content为空，或长度小于10，则跳过本次for循环
                    if not content or len(content) < 10:
                        continue
                    # 因为原文本中有很多繁体字，因此需要将繁体字转换为简体字
                    content = converter.convert(content)
                    # 添加起始和结束标记
                    marked_content = start_tag + content + end_tag
                    all_text += marked_content
                except json.JSONDecodeError:  # 如果解析JSON失败，打印错误信息
                    logging.error(f"Error decoding JSON in {file_path}: {line}")

# (4) 分块处理(从句号处切分，不小于预设的最小长度为)---------------------------------------------------------------------------
logging.info('(4) 分块处理(从句号处切分，不小于预设的最小长度为)...')
processed_texts = []  # 存储处理后的文本
min_len = 200 # 设置文本块的最小长度
start_index = 0 # 初始化起始索引
with tqdm(total=len(all_text), desc="分块处理进度") as pbar: # 使用 tqdm 包装循环，显示进度条
    while start_index + min_len < len(all_text):  # 确保还有足够的字符可以处理
        # 从当前索引开始，找到满足长度要求且以句号结尾的位置
        end_index = start_index + min_len # 计算结束索引(至少要达到min_len)
        while end_index < len(all_text):  # 只要没有到达文本末尾，就继续向后移动
            # 如果找到句号，就停止移动
            if all_text[end_index - 1] == '。': 
                break
            end_index += 1 # 否则，移动到下一个位置

        block = all_text[start_index:end_index].strip()  # 提取文本块，并去掉前后的空格
        # 如果没有以</s>结尾，就加上</s>
        if not block.endswith(end_tag):
            block = block + end_tag
        # 如果没有以<s>开头，就加上<s>
        if not block.startswith(start_tag):
            block = start_tag + block
        # 去掉文本块中的所有 <s></s>
        # 注意：<s></s>之所以会产生，是因为在分块的时候，找到的句号后面恰好是</s>，所以下一个文本块会以</s>开头，
        #      而在上一步处理时，开头会加上<s>，所以就会产生<s></s>
        while start_tag + end_tag in block:
            block = block.replace(start_tag + end_tag, "")
        # 将文本块封装成JSON对象
        processed_obj = {"text": block}
        # 将处理后的文本添加到列表中
        processed_texts.append(processed_obj)
        # 更新起始索引(以上一个块的结束位置为起始位置)
        start_index = end_index
        # 更新进度条
        pbar.update(end_index - pbar.n)

# (5) 处理之后的文本，随机抽取90%作为训练数据，10%作为验证数据------------------------------------------------------------------
# 打乱处理后的文本顺序
random.shuffle(processed_texts)

# 划分训练集和验证集
train_size = int(len(processed_texts) * 0.997)
train_data = processed_texts[:train_size]
eval_data = processed_texts[train_size:]

# (6) 保存训练和验证数据到文件----------------------------------------------------------------------------------------------
# 定义保存函数
def save_data(data, base_dir, file_prefix):
    os.makedirs(base_dir, exist_ok=True)
    file_index = 0
    lines_per_file = 500000
    for i in range(0, len(data), lines_per_file):
        file_path = os.path.join(base_dir, f"{file_prefix}_{file_index}.jsonl")
        with open(file_path, 'w', encoding='utf-8') as outfile:
            for j, obj in enumerate(data[i:i + lines_per_file]):
                if j > 0:
                    outfile.write('\n')
                json.dump(obj, outfile, ensure_ascii=False)
        file_index += 1
        logging.info(f"已保存 {file_path}")

# 保存训练数据
save_data(train_data, train_dir, "train_data")

# 保存验证数据
save_data(eval_data, eval_dir, "eval_data")

logging.info("数据处理完成并已按要求保存到训练集和验证集目录中。")
