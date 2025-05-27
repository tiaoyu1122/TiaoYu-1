# 鯈鱼(tiáoyu)
为全人类自由而奋斗！

<div align="center">

<img src="./images/logo.png" alt="logo" width="300">

</div>


## 愿景

AI时代已经来临！

随着AI技术的蓬勃发展，可以预见，简单生产活动将被AI逐渐取代，人类有可能会进入一个全新的纪元！

这将会是一个什么样的时代？

是一个生产力大爆发、物质生活极大丰富的时代，抑或是一个充满焦虑、恐慌的时代？我们暂时无法预判！

纵观历史，技术的发展总是带来生产力的突飞猛进，也总是带来人类社会的进一步发展，物质更加丰富、社会更加平等、人类更加自由。

但这次可能不同！

以往的技术进步，都是建立在“人”这个基础之上的。无论是犁，还是蒸汽机，抑或是个人计算机，都是依附于人的“工具”，也创造了更多的劳动力需求。而AI却可以在很大程度上摆脱“人”的控制，成为一个可以独立思考(虽然这种观点是存疑的，甚至可以说是伪命题的，但不可否认它的行为很像独立思考)，并完成复杂任务的主体。这意味着，社会生产活动的不再必须基于大量的劳动力操纵工具，而是由少数人维护AI来开发、利用、消费自然资源。

一个极其卑鄙的想法随之而来：这个时代的人类社会，可以由少量精英、庞大的AI、有限的自然资源组成！而大多数“人”，可能是冗余的！尤其是这些冗余的人，是作为有限自然资源的分母存在！

这个时候，人类社会必将面临着一个可怕的抉择！而这个抉择很可能是由精英阶层做出的！

所以，大多数的人有必要思考一个问题：我的命运将会被怎样安排？

但在这里，我们想直接提出一个质疑：难道因为有人掌握了比我们更聪明、更先进的技术，然后就理所当然地成为了一个统治阶级？答案当然是否定的。“人生而平等”这句话既不是一句宗教口号，也不是一个由教育灌输到人脑中的信仰，而是一个钢铁般的社会法则！人类的未来不应该由少数精英决定！

那如何才能实现“人人平等”呢？要知道，平等从来不是靠别人施舍得来的，而是通过自身强大争取来的！在新的时代，只有将AI技术普及化，使其成为一个人人都可以制造并使用的工具，让人与人之间不再具有技术上的鸿沟，就能在一定程度上避免技术掌握在少数人手中而产生新形式的剥削与压迫，也就不会有“一部分人的生存要仰仗另一部分人的怜悯”之类的事情发生。只有每个人手上都持有一把剑，他们才会围坐下来讲道理！

因此，本项目将致力于普及AI技术！我们将用最直接、最容易理解的方式，来介绍AI技术，使得绝大部分接受过基础教育的人都能够掌握它！

也正是因为这个原因，我们将项目命名为“鯈(tiáo)鱼(yu)”。《山海经》中记载：“彭水出焉，而西流注于芘湖之水，中多鯈鱼，其状如鸡而赤毛，三尾、六足、四首，其音如鹊，食之可以已忧。”希望这个项目可以为世人摆脱忧患尽绵薄之力！

当然，虽然本项目立意很高，但囿于作者能力和时间的限制，距实现上述目标确相去甚远。尤其是整个项目的组织逻辑、知识点覆盖度、代码通俗性等方面，还有非常大的提升空间。但只要能起些“抛砖引玉”的作用，于愿足矣！

感谢本项目所借鉴的诸多优秀开源项目(包括但不限于minimind、Steel_LLM、llama3等)，向这些项目的贡献者表达崇高的敬意！

**致敬人类灵魂的导师——卡尔·海因里希·马克思！**

**致敬全人类解放事业的统帅与舵手——毛泽东！**

## 项目特点

 - 本项目是站在巨人的肩膀之上，**参照甚至照搬**了诸多优秀开源项目(如minimind、Steel_LLM、llama3等，尤其是minimind)，并力求使代码尽可能地更加通俗易懂，更适合新手学习。
 - 本项目的几乎每一行代码(一些重复的代码除外)都添加了**注释**，详细介绍了代码的作用，方便阅读与理解。
 - 本项目基本上覆盖了常见 LLM 模型的**全部训练流程**，包括：预训练、有监督微调(SFT)、人类反馈强化学习(ELHF)、LoRA微调、推理模型训练(Reasoning)、知识蒸馏(KD)等。
 - 本项目**不提供现成的训练数据**，感兴趣的同学可以参照 script/pretrain_data_processing/pretrain_data_download.py和script/pretrain_data_processing/pretrain_data_generate.py 进行数据收集和处理。
 - 本项目对 LLM 模型的一些**关键知识点**进行了总结和梳理(借鉴和照搬了网络上的公开资料)，详见 notebook 下的 markdown 文件。
 - 本项目**仅供学习交流之用，承诺不用于商业用途**。

## 项目内容
- [LLM相关知识点总结](notebook)
- [模型代码](script)
    - [预训练数据处理和加载](script/pretrain_data_processing)
    - [SFT数据加载](script/sft_data_processing)
    - [DPO数据加载](script/dpo_data_processing)
    - [分词器模型训练](script/tokenizer_training/BPE_tokenizer_training.py)
    - [模型结构](script/modeling)
    - [模型训练](script/training)

## 推荐阅读顺序

[相关概念(文档)](notebook/1-相关概念.md) -> 
[模型构建(文档)](notebook/2-模型构建.md) -> 
[(粗读)模型超参数(代码)](script/modeling/model_config.py) -> 
[(可选)预训练数据下载(代码)](script/pretrain_data_processing/pretrain_data_download.py) ->
[(可选)预训练数据处理(代码)](script/pretrain_data_processing/pretrain_data_generate.py) -> 
[分词器(文档)](notebook/3-分词器.md) -> 
[分词器(代码)](script/tokenizer_training/BPE_tokenizer_training.py) -> 
[(可选)分词器模型(json文件)](model/BPE_tokenizer/) -> 
[LLM模型整体结构(代码)](script/modeling/model.py) -> 
[模型超参数(代码)](script/modeling/model_config.py) -> 
[嵌入层(文档)](notebook/4-嵌入层.md) ->
[正则化(文档)](notebook/5-正则化.md) ->
[位置编码(文档)](notebook/6-位置编码.md) ->
[位置编码(代码)](script/modeling/model_utils/RoPE.py) ->
[归一化(文档)](notebook/7-归一化.md) ->
[归一化(代码)](script/modeling/model_utils/Normalization.py) ->
[线性层(文档)](notebook/8-线性层.md) ->
[激活函数(文档)](notebook/13-激活函数.md) ->
[解码器(文档)](notebook/9-解码器模块.md) ->
[解码器(代码)](script/modeling/model_utils/DecoderBlock.py)
[多头掩码自注意力机制(文档)](notebook/10-多头掩码自注意力机制.md) ->
[多头掩码自注意力机制(代码)](script/modeling/model_utils/Attention.py) ->
[Flash Attention(文档)](notebook/22-FlashAttention.md) ->
[MOE前馈神经网络(文档)](notebook/11-MOE前馈神经网络.md) ->
[MOE前馈神经网络(代码)](script/modeling/model_utils/MoE.py) ->
[输出结果类(代码)](script/modeling/model_utils/Output.py) ->
[预训练数据加载(代码)](script/pretrain_data_processing/pretrain_data_load.py) ->
[交叉熵损失函数(文档)](notebook/12-交叉熵损失函数.md) ->
[信息量、熵、交叉熵、KL散度等(文档)](notebook/17-信息量、熵、交叉熵、KL散度等.md) ->
[优化器(文档)](notebook/14-优化器.md) ->
[(复习)模型构建(文档)](notebook/2-模型构建.md) -> 
[预训练(代码)](script/training/train_pretrain.py) ->
[SFT(代码)](script/training/train_sft.py) ->
[人类反馈强化学习(文档)](notebook/20-人类反馈强化学习.md) ->
[人类反馈强化学习(代码)](script/training/train_rlhf.py) ->
[LoRA微调(代码)](script/training/train_lora.py) ->
[知识蒸馏(代码)](script/training/train_kd.py) ->
[推理模型训练(代码)](script/training/train_reasoning.py)
[(可选)分类模型评价指标(文档)](notebook/15-分类模型评价指标.md) ->
[(可选)梯度消失与梯度爆炸(文档)](notebook/18-梯度消失与梯度爆炸.md) ->
[(可选)非极大抑制算法(文档)](notebook/19-非极大抑制算法.md) ->
[(可选)GPT和BERT(文档)](notebook/21-GPT和BERT.md) ->
[(可选)Q-Former(文档)](notebook/23-Q-Former.md)

**喜欢的小伙伴可以点个 🌟，以促进 LLM 技术的进一步普及，谢谢！**