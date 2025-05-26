import os
import sys
current_dir = os.path.dirname(os.path.abspath(__file__))  # 获取当前脚本所在的目录
parent_dir = os.path.dirname(current_dir)                 # 获取上一级目录
sys.path.append(parent_dir)                               # 将上一级目录添加到 sys.path 中，以便导入其他模块
os.environ["TOKENIZERS_PARALLELISM"] = "false"            # 禁用并行处理
import time
import math
import warnings
import datetime
import torch
from torch import nn                                        # 导入PyTorch的神经网络模块
import torch.nn.functional as F
from contextlib import nullcontext
from torch import optim
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
from modeling.model_config import TiaoyuConfig
from modeling.model import TIAOYU
from sft_data_processing.sft_data_load import SFTDataset


if __name__ == "__main__":
    
    torch.autograd.set_detect_anomaly(True)
    warnings.filterwarnings('ignore')

    print("开始白盒蒸馏(KD)TIAOYU模型...")

    # 1. 初始化TIAOYU模型超参数配置对象 
    print("1. 初始化TIAOYU模型超参数...")
    print("(1) 初始化“教师”模型超参数...")
    teacher_config = TiaoyuConfig() # 这里使用TiaoyuConfig类中的默认配置参数
    print("(2) 初始化“学生”模型超参数...")
    student_config = TiaoyuConfig(embed_dim=256)
    
    # 2. 设置模型预训练参数
    print("2. 设置模型预训练参数...")
    pretrain_model_path = "./model/reasoning_model/reasoning_512.pth" # 预训练模型路径
    output_dir = "./model/kd_model"                                   # 模型保存路径
    os.makedirs(output_dir, exist_ok=True)
    logging_dir = "./model/kd_model/logs"                             # 训练日志保存路径
    os.makedirs(logging_dir, exist_ok=True)
    logging_file = os.path.join(logging_dir, f"kd_log_{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}.txt")  # 训练日志文件路径，时间后缀改为日期-小时-分钟-秒格式
    train_data_dir = "./data/sft_data/train_data/sft.jsonl"      # 训练数据路径
    eval_data_dir = None                                         # 验证数据路径
    device_type = "gpu" if torch.cuda.is_available() else "cpu"  # 训练设备类型(GPU或CPU)
    epochs = 6                                                   # 训练轮数
    batch_size =  16                                             # 批大小
    learning_rate = 5e-6                                         # 学习率
    dtype = "bfloat16"                                           # 数据类型
    alpha = 0.2                                                  # 交叉熵损失权重
    dataload_worker_num = 1                                      # 数据加载工作线程数
    accumulation_steps = 16                                      # 梯度累积步数
    grad_clip = 1.0                                              # 梯度裁剪阈值
    log_interval = 100                                            # 日志记录间隔
    save_interval = 2000                                         # 模型保存间隔
    sft_config = \
        f"""模型训练超参数:
            (1) 预训练模型路径: {pretrain_model_path} \n \
            (2) 蒸馏模型将被保存到: {output_dir} \n \
            (3) 蒸馏日志将被保存到: {logging_file}\n \
            (4) 训练数据路径: {train_data_dir}\n \
            (5) 验证数据路径: {eval_data_dir}\n \
            (6) 训练设备类型: {device_type}\n \
            (7) 训练轮数: {epochs}\n \
            (8) 批大小: {batch_size}\n \
            (9) 学习率: {learning_rate}\n \
            (10) 数据类型: {dtype}\n \
            (11) 交叉熵损失权重: {alpha}\n \
            (12) 数据加载工作线程数: {dataload_worker_num}\n \
            (13) 梯度累积步数: {accumulation_steps}\n \
            (14) 梯度裁剪阈值: {grad_clip}\n \
            (15) 日志记录间隔: {log_interval}\n \
            (16) 模型保存间隔: {save_interval}\n"""
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
    print("(1) 初始化教师模型...")
    teacher_model = TIAOYU(teacher_config)
    teacher_dict = torch.load(pretrain_model_path, map_location=device)
    teacher_model.load_state_dict(teacher_dict, strict=False)
    teacher_model = teacher_model.to(device)
    print(f"教师模型结构: {teacher_model}")
    print(f"教师模型的参数量: {sum(p.numel() for p in teacher_model.parameters() if p.requires_grad) / 1e6:.3f} 百万")
    with open(logging_file, "a", encoding="utf-8") as f:        # 写入模型训练超参数配置到日志文件中
        f.write(f'教师模型结构: {teacher_model} \n')
        f.write(f"教师模型的可训练参数量: {sum(p.numel() for p in teacher_model.parameters() if p.requires_grad)} \n")
    teacher_model.eval()
    teacher_model.requires_grad_(False)
    
    print("(2) 初始化学生模型...")
    student_model = TIAOYU(student_config).to(device)
    # 自定义初始化方法
    def init_weights(m):
        if isinstance(m, nn.Linear):
            torch.nn.init.kaiming_uniform_(m.weight, nonlinearity='relu')
            if m.bias is not None:
                torch.nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Embedding):
            torch.nn.init.normal_(m.weight, mean=0, std=0.01)
        elif isinstance(m, nn.LayerNorm):
            nn.init.zeros_(m.bias)
            nn.init.ones_(m.weight)
    student_model.apply(init_weights)
    print(f"学生模型结构: {student_model}")
    print(f"学生模型的参数量: {sum(p.numel() for p in student_model.parameters() if p.requires_grad) / 1e6:.3f} 百万")
    with open(logging_file, "a", encoding="utf-8") as f:        # 写入模型训练超参数配置到日志文件中
        f.write(f'学生模型结构: {student_model} \n')
        f.write(f"学生模型的可训练参数量: {sum(p.numel() for p in student_model.parameters() if p.requires_grad)} \n")

    print("(3) 初始化分词器...")
    BPE_tokenizer = AutoTokenizer.from_pretrained("./model/BPE_tokenizer")

    # 5. 创建用于训练和验证的数据加载器
    print("5. 创建用于训练和验证的数据加载器...")
    print("(1) 创建训练数据加载器...")
    # 实例化自定义的数据集类，用于加载和预处理预训练任务所需的训练数据
    train_ds = SFTDataset(data_path=train_data_dir, tokenizer=BPE_tokenizer, max_seq_len=student_config.max_seq_len)
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
    optimizer = optim.AdamW(student_model.parameters(), lr=learning_rate)
    
    # 7. 定义一个KL损失函数
    print('7. 定义损失函数...')
    def loss_fun(student_logits, teacher_logits, temperature=1.0, reduction='batchmean'):
        with torch.no_grad():
            teacher_probs = F.softmax(teacher_logits / temperature, dim=-1).detach()
        student_log_probs = F.log_softmax(student_logits / temperature, dim=-1)
        kl = F.kl_div(student_log_probs, teacher_probs, reduction=reduction)
        return (temperature ** 2) * kl

    # 提前计算学习率调整所需的常量
    iter_per_epoch = len(train_loader)
    total_steps = epochs * iter_per_epoch

    for epoch in range(epochs):
        
        # 记录训练开始的时间
        start_time = time.time()
        student_model.train()  # 设置模型为训练模式
    
        for step, (X, Y, loss_mask) in enumerate(train_loader):
            X = X.to(device)
            Y = Y.to(device)
            loss_mask = loss_mask.to(device)
            current_step = epoch * iter_per_epoch + step
            lr = learning_rate / 10 + 0.5 * learning_rate * (1 + math.cos(math.pi * current_step / total_steps))
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr

            # 学生模型前向传播
            with ctx:
                student_Y_hat = student_model(X)
                student_logits = student_Y_hat.logits

            # 教师模型前向传播
            teacher_logits = teacher_model(X).logits
            vocab_size_student = student_logits.size(-1)
            teacher_logits = teacher_logits[..., :vocab_size_student]

            # 计算损失 -----------------------------------------------------------------------------------------------
            loss_mask_flat = loss_mask.view(-1)

            # 1) 学生模型预测结果 Y_hat 与 Y 之间的交叉熵损失
            if alpha != 0.0:
                ce_loss = F.cross_entropy(student_logits.view(-1, student_logits.size(-1)),
                                          Y.view(-1), 
                                          ignore_index=0,
                                          reduction='none')
                ce_loss = torch.sum(ce_loss * loss_mask_flat) / loss_mask_flat.sum() + student_Y_hat.aux_loss
            else:
                ce_loss = 0.0

            # 2) 学生模型预测结果与教师模型预测结果之间的蒸馏损失
            # 只在有效token位置做蒸馏
            kd_loss = loss_fun(student_logits.view(-1, student_logits.size(-1))[loss_mask_flat == 1],
                               teacher_logits.view(-1, teacher_logits.size(-1))[loss_mask_flat == 1],
                               temperature=1.0)
            
            # 3) 总损失 = alpha * cd_loss + (1-alpha) * kd_loss
            loss = (alpha * ce_loss + (1 - alpha) * kd_loss) / accumulation_steps
            # ----------------------------------------------------------------------------------------------- 计算完成

            scaler.scale(loss).backward()

            if (step + 1) % accumulation_steps == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(student_model.parameters(), grad_clip)

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
                student_model.eval()
                ckp = f'{output_dir}/kd_{student_config.embed_dim}.pth'
                if isinstance(student_model, torch.nn.parallel.DistributedDataParallel):
                    state_dict = student_model.module.state_dict()
                else:
                    state_dict = student_model.state_dict()
                torch.save(state_dict, ckp)
                student_model.train()
