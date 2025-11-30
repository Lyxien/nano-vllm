from dataclasses import dataclass
import torch


# 这个类的目的其实就是 设置一些全局变量（本质和传统的上下文的概念有点区别）
# 避免了将变量通过层层传递的方式进行使用
# 一个 batch 对应一个 context，内部包含每条 sequence 的状态信息
# 例如 cu_seqlens_q = [0, 5, 9] 表示 当前 batch 中：
#   第一条 sequence 的 query token 在拼接后是 [0, 5)，长度 = 5
#   第二条 sequence 的 query token 在拼接后是 [5, 9)，长度 = 4
@dataclass
class Context:
    # 是否是 prefill 阶段
    is_prefill: bool = False
    # Query 的累计长度（cumulative sequence lengths），通常是 prefix sum，用于把多序列拼接成一个 batch。
    cu_seqlens_q: torch.Tensor | None = None
    # Key 的累计长度，和 cu_seqlens_q 类似，但可能不同，因为 key/value cache 可以增长（生成阶段）。
    cu_seqlens_k: torch.Tensor | None = None
    # 当前 batch 中 query 的最大长度，用于 allocation 或 masking。
    max_seqlen_q: int = 0
    # 当前 batch 中 key 的最大长度，主要影响 KV cache。
    max_seqlen_k: int = 0
    # batch 中 token/sequence 对应的 slot 映射，用于对齐不同序列的 KV cache。
    slot_mapping: torch.Tensor | None = None
    # 每个 sequence 的上下文长度，用于 tracking 每条 sequence 在 batch 中的位置。
    context_lens: torch.Tensor | None = None
    # KV cache block table，对应每个 sequence 的 block 状态，用于加速注意力计算。
    block_tables: torch.Tensor | None = None

_CONTEXT = Context()

def get_context():
    return _CONTEXT

def set_context(is_prefill, cu_seqlens_q=None, cu_seqlens_k=None, max_seqlen_q=0, max_seqlen_k=0, slot_mapping=None, context_lens=None, block_tables=None):
    global _CONTEXT
    _CONTEXT = Context(is_prefill, cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k, slot_mapping, context_lens, block_tables)

def reset_context():
    global _CONTEXT
    _CONTEXT = Context()
