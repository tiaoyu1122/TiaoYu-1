import torch
from torch import nn


# 定义Lora网络结构
class LoRA(nn.Module):
    def __init__(self, in_features, out_features, rank):
        super().__init__()
        self.rank = rank  # LoRA的秩（rank），控制低秩矩阵的大小
        self.A = nn.Linear(in_features, rank, bias=False)   # 低秩矩阵A
        self.B = nn.Linear(rank, out_features, bias=False)  # 低秩矩阵B
        # 矩阵A高斯初始化
        self.A.weight.data.normal_(mean=0.0, std=0.02)
        # 矩阵B全0初始化
        self.B.weight.data.zero_()

    def forward(self, x):
        return self.B(self.A(x))


def apply_lora(model, rank=8):
    for name, module in model.named_modules():
        # if isinstance(module, nn.Linear) and module.weight.shape[0] == module.weight.shape[1]:
        if isinstance(module, nn.Linear) and module.weight.shape[0] >= 128 and module.weight.shape[1] >= 128:
            # nn.Linear 实现了操作：y = xA^T + b
            # module.weight.shape[0] 是 out_features，也就是输出维度
            # module.weight.shape[1] 是 in_features，也就是输入维度
            lora = LoRA(in_features=module.weight.shape[1], 
                        out_features=module.weight.shape[0], 
                        rank=rank).to(model.device)
            setattr(module, "lora", lora)
            original_forward = module.forward

            # 显式绑定
            def forward_with_lora(x, layer1=original_forward, layer2=lora):
                return layer1(x) + layer2(x)

            module.forward = forward_with_lora


def load_lora(model, path):
    state_dict = torch.load(path, map_location=model.device)
    for name, module in model.named_modules():
        if hasattr(module, 'lora'):
            lora_state = {k.replace(f'{name}.lora.', ''): v for k, v in state_dict.items() if f'{name}.lora.' in k}
            module.lora.load_state_dict(lora_state)


def save_lora(model, path):
    state_dict = {}
    for name, module in model.named_modules():
        if hasattr(module, 'lora'):
            lora_state = {f'{name}.lora.{k}': v for k, v in module.lora.state_dict().items()}
            state_dict.update(lora_state)
    torch.save(state_dict, path)


if __name__ == '__main__':
    import os
    import sys
    current_dir = os.path.dirname(os.path.abspath(__file__))  # 获取当前脚本所在的目录
    parent_dir = os.path.dirname(current_dir)                 # 获取上一级目录
    sys.path.append(parent_dir)                               # 将上一级目录添加到 sys.path 中，以便导入其他模块
    os.environ["TOKENIZERS_PARALLELISM"] = "false"            # 禁用并行处理
    from modeling.model_config import TiaoyuConfig
    from modeling.model import TIAOYU

    tiaoyu_config = TiaoyuConfig() # 这里使用TiaoyuConfig类中的默认配置参数
    tiaoyu_model = TIAOYU(tiaoyu_config)
    print(f"TIAOYU模型结构: {tiaoyu_model}")
    apply_lora(tiaoyu_model, rank= 8) # 应用LoRA
    print(f"LORA模型结构: {tiaoyu_model}")

    # test_linear = nn.Linear(128, 64)
    # print(f'test_linear输入维度: {test_linear.weight.shape[1]}')
    # print(f'test_linear输出维度: {test_linear.weight.shape[0]}')