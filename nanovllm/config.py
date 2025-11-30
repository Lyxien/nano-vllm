import os
from dataclasses import dataclass
from transformers import AutoConfig


@dataclass
class Config:
    model: str
    # 一次推理过程中当前批次中正在被模型计算的 token 总数量上限，包括输入和输出token
    max_num_batched_tokens: int = 16384
    # 最大并发 sequence 数
    # sequence = token 序列（记录token生成到哪里了）+ 管理 KV Cache （cache 所在的block） 等
    # 一般情况下， 1个请求（业务层概念） 相当于 1个sequence（模型层概念）
    max_num_seqs: int = 512
    # 一条序列（一个对话/一个生成请求）能包含的 token 最大数量（包含了输入和输出的token）
    max_model_len: int = 4096
    # 显存使用比例（kv cache 的大小是灵活可变的）
    gpu_memory_utilization: float = 0.9
    # 多GPU张量并行数量
    tensor_parallel_size: int = 1
    # 是否强制使用 eager 模式，即关闭编译优化
    enforce_eager: bool = False
    # HuggingFace 的模型配置（HuggingFace Transformers 框架里的模型配置对象，用于描述模型信息，如每层 hidden size 大小、attention head 数目）
    hf_config: AutoConfig | None = None
    # 结束符的 token id，生成 token 当中如果发现生成的 终止符 那么意味着 decoder 结束
    eos: int = -1
    # kvcache 的 block 大小
    # 每个 token 会产生 K: [num_layers, num_heads, head_dim] 和 V: [num_layers, num_heads, head_dim]
    # 每 kvcache_block_size 个 token 的 K 和 V 组成一个 block 
    kvcache_block_size: int = 256
    #  kvcache_block 的数量，-1 表示根据 gpu_memory_utilization 和 实际显存占用 自动计算
    num_kvcache_blocks: int = -1

    # 对配置进行检查 
    def __post_init__(self):
        assert os.path.isdir(self.model)
        assert self.kvcache_block_size % 256 == 0
        assert 1 <= self.tensor_parallel_size <= 8
        # 自动读取配置文件 config.json（transformers库中提供实现） 
        self.hf_config = AutoConfig.from_pretrained(self.model)
        self.max_model_len = min(self.max_model_len, self.hf_config.max_position_embeddings)
        assert self.max_num_batched_tokens >= self.max_model_len
