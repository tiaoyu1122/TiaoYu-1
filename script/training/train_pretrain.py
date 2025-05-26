import sys
import os
current_dir = os.path.dirname(os.path.abspath(__file__))  # 获取当前脚本所在的目录
parent_dir = os.path.dirname(current_dir)                 # 获取上一级目录
sys.path.append(parent_dir)                               # 将上一级目录添加到 sys.path 中，以便导入其他模块
os.environ["TOKENIZERS_PARALLELISM"] = "false"            # 禁用并行处理
import time
import datetime
import math
import warnings
import torch
from torch import optim, nn
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from contextlib import nullcontext
from transformers import AutoTokenizer
from modeling.model_config import TiaoyuConfig
from modeling.model import TIAOYU
from pretrain_data_processing.pretrain_data_load import PretrainDataset

if __name__ == '__main__':

    warnings.filterwarnings('ignore')

    print("开始预训练TIAOYU模型...")

    # 1. 初始化TIAOYU模型超参数配置对象 
    print("1. 初始化TIAOYU模型超参数...")
    tiaoyu_config = TiaoyuConfig() # 这里使用TiaoyuConfig类中的默认配置参数

    # 2. 设置模型预训练参数
    print("2. 设置模型预训练参数...")
    output_dir = "./model/pretrain_model"                        # 模型保存路径
    os.makedirs(output_dir, exist_ok=True)
    logging_dir = "./model/pretrain_model/logs"                  # 训练日志保存路径
    os.makedirs(logging_dir, exist_ok=True)
    logging_file = os.path.join(logging_dir, f"pretrain_log_{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}.txt")  # 训练日志文件路径，时间后缀改为日期-小时-分钟-秒格式
    train_data_dir = "./data/pretrain_data/train_data/pretrain_hq.jsonl"          # 训练数据路径
    eval_data_dir = "./data/pretrain_data/eval_data"             # 验证数据路径
    device_type = "gpu" if torch.cuda.is_available() else "cpu"  # 训练设备类型(GPU或CPU)
    epochs = 3                                                   # 训练轮数
    batch_size =  16                                             # 批大小
    learning_rate = 5e-4                                         # 学习率
    dtype = "bfloat16"                                           # 数据类型
    dataload_worker_num = 1                                      # 数据加载工作线程数
    accumulation_steps = 8                                       # 梯度累积步数
    grad_clip = 1.0                                              # 梯度裁剪阈值
    log_interval = 100                                           # 日志记录间隔
    save_interval = 500                                          # 模型保存间隔
    pretrain_config = \
        f"""模型训练超参数:
            (1) 预训练模型将被保存到: {output_dir} \n \
            (2) 预训练日志将被保存到: {logging_file}\n \
            (3) 训练数据路径: {train_data_dir}\n \
            (4) 验证数据路径: {eval_data_dir}\n \
            (5) 训练设备类型: {device_type}\n \
            (6) 训练轮数: {epochs}\n \
            (7) 批大小: {batch_size}\n \
            (8) 学习率: {learning_rate}\n \
            (9) 数据类型: {dtype}\n \
            (10) 数据加载工作线程数: {dataload_worker_num}\n \
            (11) 梯度累积步数: {accumulation_steps}\n \
            (12) 梯度裁剪阈值: {grad_clip}\n \
            (13) 日志记录间隔: {log_interval}\n \
            (14) 模型保存间隔: {save_interval}\n"""
    print(pretrain_config)
    # with open(logging_file, "a", encoding="utf-8") as f:
    #     f.write(pretrain_config) # 写入模型训练超参数配置到日志文件中

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
    tiaoyu_model = TIAOYU(tiaoyu_config).to(device)

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

    tiaoyu_model.apply(init_weights)

    print(f"TIAOYU模型结构: {tiaoyu_model}")
    print(f"模型的可训练参数量: {sum(p.numel() for p in tiaoyu_model.parameters() if p.requires_grad) / 1e6:.3f} 百万")
    # with open(logging_file, "a", encoding="utf-8") as f: # 写入模型训练超参数配置到日志文件中
    #     f.write(f'模型结构: {tiaoyu_model} \n')
    #     f.write(f"模型的可训练参数量: {sum(p.numel() for p in tiaoyu_model.parameters() if p.requires_grad) / 1e6:.3f} \n")
    print("(2) 初始化分词器...")
    BPE_tokenizer = AutoTokenizer.from_pretrained("./model/BPE_tokenizer")
    
    # 5. 创建用于训练和验证的数据加载器
    print('5. 创建用于训练和验证的数据加载器...')
    # 实例化自定义的数据集类，用于加载和预处理预训练任务所需的训练数据
    train_ds = PretrainDataset(data_path=train_data_dir, tokenizer=BPE_tokenizer, max_seq_len=tiaoyu_config.max_seq_len)
    train_data_loader = DataLoader(dataset = train_ds, batch_size=batch_size, pin_memory=True, drop_last=False,
                                   shuffle=False, num_workers=dataload_worker_num, sampler=None)
    # 实例化自定义的数据集类，用于加载和预处理预训练任务所需的验证数据
    eval_ds = PretrainDataset(data_path=eval_data_dir, tokenizer=BPE_tokenizer, max_seq_len=tiaoyu_config.max_seq_len)
    eval_data_loader = DataLoader(dataset = eval_ds, batch_size=batch_size, pin_memory=True, drop_last=False, 
                              shuffle=False, num_workers=dataload_worker_num, sampler=None)

    # 6. 初始化优化器(指定了优化器需要更新的参数、学习率)
    print('6. 初始化优化器...')
    optimizer = optim.AdamW(tiaoyu_model.parameters(), lr=learning_rate, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.01)
    # optimizer = optim.Adam(tiaoyu_model.parameters(), lr=learning_rate, betas=(0.9, 0.999), eps=1e-8)
    # 定义学习率调度器(例如余弦退火)
    # scheduler = CosineAnnealingLR(optimizer, T_max=10)  # T_max 是调度周期

    # 7. 定义一个交叉熵损失函数
    # reduction='none'表示不进行任何归约，即返回每个样本的损失，而不是平均损失
    # 因为后面还要结合loss_mask进行加权求和，所以需要保留每个样本的损失
    print('7. 定义损失函数...')
    loss_fun = nn.CrossEntropyLoss(reduction='none')

    # 提前计算学习率调整所需的常量
    iter_per_epoch = len(train_data_loader)
    total_steps = epochs * iter_per_epoch

    # 8. 开始训练
    print('8. 开始训练...')
    # 遍历每个epoch进行模型训练
    for epoch in range(epochs):
        # (1) 记录训练开始的时间
        start_time = time.time()
        tiaoyu_model.train()  # 设置模型为训练模式
        
        # 循环遍历训练数据集，获取数据X、Y和loss_mask
        for step, (X, Y, loss_mask) in enumerate(train_data_loader):

            # (2) 将数据移动到指定的设备上
            X = X.to(device)
            Y = Y.to(device)
            loss_mask = loss_mask.to(device)
            
            # (3) 更新学习率，调整optimizer中的学习率参数
            current_step = epoch * iter_per_epoch + step
            lr = learning_rate / 10 + 0.5 * learning_rate * (1 + math.cos(math.pi * current_step / total_steps))
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr
            
            # (4) 模型前向传播
            with ctx:  # 上下文管理器，用于控制精度模式. 如果是GPU，则使用torch.cuda.amp.autocast()，否则使用nullcontext()
                # i. 将X输入模型，计算模型预测结果
                Y_hat = tiaoyu_model(X)
                # ii. 根据Y_hat.logits和Y计算损失
                loss = loss_fun(Y_hat.logits.view(-1, Y_hat.logits.size(-1)), Y.view(-1)).view(Y.size())
                # iii. 使用loss_mask对损失进行加权，并计算平均损失，并加上辅助损失Y_hat.aux_loss
                loss = (loss * loss_mask).sum() / (loss_mask.sum() + 1e-8) + Y_hat.aux_loss
                # iv. 将损失除以梯度累积步数accumulation_steps
                loss = loss / accumulation_steps
            # (5) 梯度缩放(作用详见3(3)，解决“使用不同精度的浮点数（如FP16和FP32），可能会导致梯度数值过小或过大，进而影响模型的训练效果”的问题)
            scaler.scale(loss).backward()
            
            # (6) 梯度累积与反向传播
            # 当显存不足或者为了模拟更大的batch size时，会采用梯度累积的策略。
            # 这意味着不是每次迭代都立即更新模型参数，而是将多次迭代的梯度累积起来，然后再一次性更新模型参数。
            # 下面，当累积步数达到accumulation_steps时，进行梯度裁剪、优化器步长更新、梯度缩放器更新，并重置梯度
            if (step + 1) % accumulation_steps == 0:
                try:
                    # i. 梯度反向缩放
                    # 在进行梯度累积并准备更新模型参数之前，需要先对梯度进行反向缩放，以恢复到原始的梯度大小。
                    # 这是因为在之前的步骤中，我们对损失值进行了缩放，所以计算得到的梯度也是缩放后的。
                    scaler.unscale_(optimizer)
                    # ii. 梯度裁剪
                    # 对模型参数的梯度进行裁剪，确保梯度的范数不会超过grad_clip。这是为了防止梯度爆炸，保证训练的稳定性。
                    torch.nn.utils.clip_grad_norm_(tiaoyu_model.parameters(), grad_clip)
                    # iii. 模型参数更新
                    # 调用优化器的step方法，根据计算得到的梯度更新模型参数。
                    # 在自动混合精度训练中，这一步也是通过梯度缩放器来完成的，以确保更新的步长是合适的。
                    scaler.step(optimizer)
                    # iv. 梯度缩放器更新
                    # 更新梯度缩放器，使其能够根据当前的梯度分布来调整缩放因子，以更好地适应后续的训练。
                    scaler.update()
                except RuntimeError as e:
                    print(f"Runtime error at step {step+1}: {e}. Skipping this update.")
                # v. 梯度清零
                # 清除优化器中所有参数的梯度，为下一次迭代的梯度计算做准备。
                # set_to_none=True参数的作用是将梯度设置为None而不是0，这可以减少内存开销并提高性能。
                optimizer.zero_grad(set_to_none=True)
                        
            # (7) 日志记录(每隔log_interval步记录一次)
            if (step % log_interval == 0) | ((step + 1) == iter_per_epoch):
                # i. 计算已经经过的时间
                spend_time = time.time() - start_time
                # ii. 拼接日志内容(记录当前损失、学习率、已用时间、剩余时间等)
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
                # iii. 打印日志内容
                print(log_str)
                # iv. 将上述日志写入文件中
                # with open(logging_file, 'a') as f:
                #     f.write(log_str + '\n')

            # (8) 保存模型(每隔save_interval步保存一次)
            if ((step + 1) % save_interval == 0) | ((step + 1) == iter_per_epoch):
                # i. 将模型设定为eval模式
                tiaoyu_model.eval()
                # ii. 定义模型文件路径
                model_file = f'{output_dir}/pretrain_hq_{tiaoyu_config.embed_dim}.pth'
                # iii. 提取模型参数
                # 从模型中提取出状态字典(state_dict)，其中包含了模型所有的参数信息。
                state_dict = tiaoyu_model.state_dict()
                # iv. 保存模型参数
                # 使用torch.save()函数将模型的参数保存到指定的文件路径中。
                torch.save(state_dict, model_file)
                # v. 将模型设定为train模式，以便下一次训练时能够正常使用
                tiaoyu_model.train()
        
        # # (9) 训练中验证(每个epoch结束后进行验证)
        # # i. 将模型设定为eval模式，以便进行验证
        # tiaoyu_model.eval()
        # eval_loss = 0.0
        # # ii. 禁用梯度计算，以减少内存消耗
        # with torch.no_grad():
        #     # 遍历验证数据集，获取数据X、Y和loss_mask
        #     for step, (X, Y, loss_mask) in enumerate(eval_data_loader):
        #         # iii.将数据移动到指定的设备上
        #         X = X.to(device)
        #         Y = Y.to(device)
        #         loss_mask = loss_mask.to(device)
        #         # iv. 模型前向传播
        #         with ctx:
        #             # a. 将X输入模型，计算模型预测结果
        #             Y_hat = tiaoyu_model(X)
        #             # b. 根据Y_hat.logits和Y计算损失
        #             loss = loss_fun(Y_hat.logits.view(-1, Y_hat.logits.size(-1)), Y.view(-1)).view(Y.size())
        #             # c. 使用loss_mask对损失进行加权，并计算平均损失(这里只关心主任务的损失，所以不考虑辅助损失Y_hat.aux_loss)
        #             loss = (loss * loss_mask).sum() / loss_mask.sum()
        #             # d. 将损失累加到eval_loss中
        #             eval_loss += loss.item()
        # # v. 计算验证集平均损失
        # eval_loss /= len(eval_data_loader)
        # # vi. 打印验证集平均损失
        # print(f'Epoch {epoch + 1} completed. Validation Loss: {eval_loss:.3f}')
        # # vii. 将上述日志写入文件中
        # with open(logging_file, 'a') as f:
        #     f.write(f'Epoch {epoch + 1} completed. Validation Loss: {eval_loss:.3f} \n')
        # # viii. 将模型设定为train模式，以便下一次训练时能够正常使用
        # tiaoyu_model.train()
