import random
import numpy as np
import torch
import warnings
from transformers import AutoTokenizer
from modeling.model import TIAOYU
from modeling.model_config import TiaoyuConfig

warnings.filterwarnings('ignore')


def init_model():
    tokenizer = AutoTokenizer.from_pretrained('./model/BPE_tokenizer')

    ckp = f'./model/pretrain_model/pretrain_hq_512.pth'
    tiaoyu_config = TiaoyuConfig() # 这里使用TiaoyuConfig类中的默认配置参数
    model = TIAOYU(tiaoyu_config)

    map_location = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    state_dict = torch.load(ckp, map_location=map_location)
    model.load_state_dict({k: v for k, v in state_dict.items() if 'mask' not in k}, strict=True)
        
    print(f'MiniMind模型参数量: {sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6:.2f}M(illion)')
    return model.eval(), tokenizer


# 设置可复现的随机种子
def setup_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def main():
    
    model, tokenizer = init_model()

    prompts = [
        '卡尔·马克思是',
        '秦始皇的陵墓在哪里？',
        '鹅鹅鹅']
    stream=True

    test_mode = int(input('[0] 自动测试\n[1] 手动输入\n'))
    messages = []
    for idx, prompt in enumerate(prompts if test_mode == 0 else iter(lambda: input('👶: '), '')):
        setup_seed(random.randint(0, 2048))
        # setup_seed(2025)  # 如需固定每次输出则换成【固定】的随机种子
        if test_mode == 0: print(f'👶: {prompt}')

        messages.append({"role": "user", "content": prompt})

        new_prompt = tokenizer.bos_token + prompt

        answer = new_prompt
        with torch.no_grad():
            x = torch.tensor(tokenizer(new_prompt)['input_ids'], device='cpu').unsqueeze(0)
            outputs = model.generate(
                x,
                stream=stream,
                repetition_penalty=1.5,
                temperature=0.5,
                top_p=0.9,
                use_kv_cache=True
            )

            print('🤖️: ', end='')
            try:
                if not stream:
                    print(tokenizer.decode(outputs.squeeze()[x.shape[1]:].tolist(), skip_special_tokens=True), end='')
                else:
                    history_idx = 0
                    for y in outputs:
                        answer = tokenizer.decode(y[0].tolist(), skip_special_tokens=True)
                        if (answer and answer[-1] == '�') or not answer:
                            continue
                        print(answer[history_idx:], end='', flush=True)
                        history_idx = len(answer)
            except StopIteration:
                print("No answer")
            print('\n')

        messages.append({"role": "assistant", "content": answer})


if __name__ == "__main__":
    main()
