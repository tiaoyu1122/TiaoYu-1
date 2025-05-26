from transformers.modeling_outputs import CausalLMOutputWithPast  # 导入CausalLMOutputWithPast类

# 定义一个新的类，继承自CausalLMOutputWithPast，用于存储模型的输出结果
class TiaoyuCausalLMOutputWithPast(CausalLMOutputWithPast):
    # CausalLMOutputWithPast是Transformers库中定义的一个类, 用于那些支持利用过去隐藏状态加速后续推理步骤的模型.
    # 如在序列生成过程中, 通过缓存前向传递的部分结果以减少重复计算, 提高效率.
    # 它包含以下属性:
    # - logits: 模型输出的logits, 通常用于计算损失.
    # - past_key_values: 模型的隐藏状态, 用于缓存.
    # - hidden_states: 模型的隐藏状态, 用于缓存.
    # - attentions: 模型的注意力权重, 用于缓存.
    # 我们可以通过继承CausalLMOutputWithPast类, 自定义自己的输出类, 用于存储模型的输出结果.
    # 在原有的4个属性基础上, 添加三个自定义属性aux_loss和cache_k, cache_v
    # 其中，aux_loss是辅助损失，cache_k是缓存的key，cache_v是缓存的value,
    # cache_k 和 cache_v 是用来替换原有属性中的past_key_values的.
    # 这样, 我们在使用这些属性的时候更加方便.

    def __init__(self, 
                 logits=None,
                 past_key_values=None, 
                 hidden_states=None, 
                 attentions=None, 
                 aux_loss=None,  # 辅助损失
                 cache_k=None,   # 缓存的key
                 cache_v=None):  # 缓存的value
        # 调用父类的构造函数
        super().__init__(logits=logits, 
                         past_key_values=past_key_values, 
                         hidden_states=hidden_states, 
                         attentions=attentions)
        # 初始化自定义属性
        self.aux_loss = aux_loss
        self.cache_k = cache_k
        self.cache_v = cache_v
    
    #(1) 重写__repr__函数，确保在打印时在TiaoyuCausalLMOutputWithPast时正常输出aux_loss和cache_k, cache_v
    def __repr__(self):
        base_repr = super().__repr__().rstrip(')')
        return f'{base_repr}, aux_loss={self.aux_loss}, cache_k={self.cache_k}, cache_v={self.cache_v})'
    
    # (2) 重写__setitem__函数, 确保可以通过调用__setitem__方法设置aux_loss和cache_k, cache_v
    def __setitem__(self, key, value):
        custom_attrs = {
            'aux_loss': lambda: setattr(self, 'aux_loss', value),
            'cache_k': lambda: setattr(self, 'cache_k', value),
            'cache_v': lambda: setattr(self, 'cache_v', value)
        }
        if key in custom_attrs:
            custom_attrs[key]()
        else:
            super().__setitem__(key, value)

if __name__ == "__main__":
    output = TiaoyuCausalLMOutputWithPast()
    print(output)