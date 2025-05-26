import sys
import os
current_dir = os.path.dirname(os.path.abspath(__file__))  # 获取当前脚本所在的目录
parent_dir = os.path.dirname(current_dir)                 # 获取上一级目录
sys.path.append(parent_dir)                               # 将上一级目录添加到 sys.path 中，以便导入其他模块
os.environ["TOKENIZERS_PARALLELISM"] = "false"            # 禁用并行处理
import time
import math
import warnings
import datetime
import torch
from contextlib import nullcontext
from torch import optim, nn
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
from modeling.model_config import TiaoyuConfig
from modeling.model import TIAOYU
from sft_data_processing.sft_data_load import SFTDataset


if __name__ == '__main__':

    torch.autograd.set_detect_anomaly(True)
    warnings.filterwarnings('ignore')

    print("开始有监督微调(SFT)TIAOYU模型...")

    # 1. 初始化TIAOYU模型超参数配置对象 
    print("1. 初始化TIAOYU模型超参数...")
    tiaoyu_config = TiaoyuConfig() # 这里使用TiaoyuConfig类中的默认配置参数

    # 2. 设置模型预训练参数
    print("2. 设置模型预训练参数...")
    pretrain_model_path = "./model/pretrain_model/pretrain_hq_512.pth" # 预训练模型路径
    output_dir = "./model/sft_model"                             # 模型保存路径
    os.makedirs(output_dir, exist_ok=True)
    logging_dir = "./model/sft_model/logs"                       # 训练日志保存路径
    os.makedirs(logging_dir, exist_ok=True)
    logging_file = os.path.join(logging_dir, f"sft_log_{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}.txt")  # 训练日志文件路径，时间后缀改为日期-小时-分钟-秒格式
    train_data_dir = "./data/sft_data/train_data/sft.jsonl"      # 训练数据路径
    eval_data_dir = None                                         # 验证数据路径
    device_type = "gpu" if torch.cuda.is_available() else "cpu"  # 训练设备类型(GPU或CPU)
    epochs = 2                                                   # 训练轮数
    batch_size =  16                                             # 批大小
    learning_rate = 2e-4                                         # 学习率
    dtype = "bfloat16"                                           # 数据类型
    dataload_worker_num = 4                                      # 数据加载工作线程数
    accumulation_steps = 16                                      # 梯度累积步数
    grad_clip = 1.0                                              # 梯度裁剪阈值
    log_interval = 100                                            # 日志记录间隔
    save_interval = 2000                                         # 模型保存间隔
    sft_config = \
        f"""模型训练超参数:
            (1) 预训练模型路径: {pretrain_model_path} \n \
            (2) SFT模型将被保存到: {output_dir} \n \
            (3) SFT日志将被保存到: {logging_file}\n \
            (4) 训练数据路径: {train_data_dir}\n \
            (5) 验证数据路径: {eval_data_dir}\n \
            (6) 训练设备类型: {device_type}\n \
            (7) 训练轮数: {epochs}\n \
            (8) 批大小: {batch_size}\n \
            (9) 学习率: {learning_rate}\n \
            (10) 数据类型: {dtype}\n \
            (11) 数据加载工作线程数: {dataload_worker_num}\n \
            (12) 梯度累积步数: {accumulation_steps}\n \
            (13) 梯度裁剪阈值: {grad_clip}\n \
            (14) 日志记录间隔: {log_interval}\n \
            (15) 模型保存间隔: {save_interval}\n"""
    print(sft_config)
    with open(logging_file, "a", encoding="utf-8") as f:
        f.write(sft_config) # 写入模型训练超参数配置到日志文件中

    # 3. 其他设置
    print("3. 其他设置...")
    print("(1) 设置运行设备和PyTorch随机数种子...")
    # 确保PyTorch在进行任何依赖于随机数的操作时，其随机性是可复现的
    if device_type == "gpu":
        device = "cuda:0"
        torch.cuda.manual_seed_all(1234)
    else:
        device = "cpu"
        torch.manual_seed(1234)
    print("(2) 设置是否启用混合精度管理...")
    # 如果device_type等于"cpu"，则ctx被赋值为nullcontext()，意味着后续使用with ctx:语句时不会有任何特别的上下文管理行为。
    # 如果device_type不等于"cpu"（通常意味着使用GPU），则ctx被赋值为torch.cuda.amp.autocast()，以便在后续代码块中自动管理混合精度。
    # 自动混合精度(Automatic Mixed Precision, AMP)允许模型在训练时使用半精度浮点数(FP16)来提高性能，同时自动处理与保持数值稳定性相关的精度问题，必要时回退到全精度浮点数(FP32)。
    # autocast()上下文管理器自动管理这些精度的转换，使得开发者无需手动管理
    ctx = nullcontext() if device_type == "cpu" else torch.amp.autocast('cuda')
    print("(3) 初始化自动混合精度(AMP)的GradScaler对象...")
    # 自动混合精度允许模型在训练时使用半精度浮点数（float16或bfloat16），这可以显著减少内存使用并提高训练速度，但同时可能会增加数值不稳定性和梯度消失/爆炸的风险。
    # GradScaler通过动态地调整梯度的缩放因子来帮助缓解这些问题。
    scaler = torch.amp.GradScaler(
        'cuda',
        enabled=(dtype in ['float16', 'bfloat16'])
    )

    # 4. 初始化 模型model 和 分词器tokenizer
    print('4. 初始化 模型 和 分词器...')
    print("(1) 初始化TIAOYU模型...")
    tiaoyu_model = TIAOYU(tiaoyu_config)
    state_dict = torch.load(pretrain_model_path, map_location=device)
    tiaoyu_model.load_state_dict(state_dict, strict=False)
    tiaoyu_model = tiaoyu_model.to(device)
    print(f"TIAOYU模型结构: {tiaoyu_model}")
    print(f"模型的可训练参数量: {sum(p.numel() for p in tiaoyu_model.parameters() if p.requires_grad) / 1e6:.3f} 百万")
    with open(logging_file, "a", encoding="utf-8") as f:        # 写入模型训练超参数配置到日志文件中
        f.write(f'模型结构: {tiaoyu_model} \n')
        f.write(f"模型的可训练参数量: {sum(p.numel() for p in tiaoyu_model.parameters() if p.requires_grad)} \n")
    
    print("(2) 初始化分词器...")
    BPE_tokenizer = AutoTokenizer.from_pretrained("./model/BPE_tokenizer")
    
    # 5. 创建用于训练和验证的数据加载器
    print("5. 创建用于训练和验证的数据加载器...")
    print("(1) 创建训练数据加载器...")
    # 实例化自定义的数据集类，用于加载和预处理预训练任务所需的训练数据
    train_ds = SFTDataset(data_path=train_data_dir, tokenizer=BPE_tokenizer, max_seq_len=tiaoyu_config.max_seq_len)
    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        pin_memory=True,
        drop_last=False,
        shuffle=False,
        num_workers=dataload_worker_num,
        sampler=None
    )
    
    # 6. 初始化优化器(指定了优化器需要更新的参数、学习率)
    print('6. 初始化优化器...')
    optimizer = optim.AdamW(tiaoyu_model.parameters(), lr=learning_rate)

    # 7. 定义一个交叉熵损失函数
    # reduction='none'表示不进行任何归约，即返回每个样本的损失，而不是平均损失
    # 因为后面还要结合loss_mask进行加权求和，所以需要保留每个样本的损失
    print('7. 定义损失函数...')
    loss_fun = nn.CrossEntropyLoss(reduction='none')
    
    # 提前计算学习率调整所需的常量
    iter_per_epoch = len(train_loader)
    total_steps = epochs * iter_per_epoch

    # 8. 开始训练
    for epoch in range(epochs):

        # 记录训练开始的时间
        start_time = time.time()
        tiaoyu_model.train()  # 设置模型为训练模式

        start_time = time.time()
        for step, (X, Y, loss_mask) in enumerate(train_loader):
            X = X.to(device)
            Y = Y.to(device)
            loss_mask = loss_mask.to(device)
            current_step = epoch * iter_per_epoch + step
            lr = learning_rate / 10 + 0.5 * learning_rate * (1 + math.cos(math.pi * current_step / total_steps))
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr

            with ctx:
                Y_hat = tiaoyu_model(X)
                loss = loss_fun(Y_hat.logits.view(-1, Y_hat.logits.size(-1)), Y.view(-1)).view(Y.size())
                loss = (loss * loss_mask).sum() / (loss_mask.sum() + 1e-8) + Y_hat.aux_loss
                loss = loss / accumulation_steps

            scaler.scale(loss).backward()

            if (step + 1) % accumulation_steps == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(tiaoyu_model.parameters(), grad_clip)

                scaler.step(optimizer)
                scaler.update()

                optimizer.zero_grad(set_to_none=True)

            if step % log_interval == 0:
                spend_time = time.time() - start_time
                log_str = \
                    '{} ~ Epoch:[{}/{}]({}/{}) Train Loss:{:.3f} lr:{:.12f} Spend Time:{}min Remaining Time:{}min'.format(
                        time.strftime("%H:%M:%S", time.localtime(time.time())),
                        epoch + 1,
                        epochs,
                        step,
                        iter_per_epoch,
                        loss.item() * accumulation_steps,
                        optimizer.param_groups[0]['lr'],
                        spend_time // 60,
                        spend_time / (step + 1) * iter_per_epoch // 60 - spend_time // 60)
                print(log_str)
                with open(logging_file, 'a') as f:
                    f.write(log_str + '\n')

            if ((step + 1) % save_interval) == 0 | ((step + 1) == iter_per_epoch):
                tiaoyu_model.eval()
                ckp = f'{output_dir}/sft_{tiaoyu_config.embed_dim}.pth'
                if isinstance(tiaoyu_model, torch.nn.parallel.DistributedDataParallel):
                    state_dict = tiaoyu_model.module.state_dict()
                else:
                    state_dict = tiaoyu_model.state_dict()
                torch.save(state_dict, ckp)
                tiaoyu_model.train()
