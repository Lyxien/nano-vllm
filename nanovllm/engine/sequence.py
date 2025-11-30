from copy import copy
from enum import Enum, auto
from itertools import count

from nanovllm.sampling_params import SamplingParams


# 当前 sequence 的计算状态
class SequenceStatus(Enum):
    # 等待状态，例如，刚插入到队列中，还没开始计算，当前服务在处理别的请求
    WAITING = auto()
    # 运行状态，正在计算当前 sequence
    RUNNING = auto()
    # 结束状态，输出碰到 结束符eos 或者 达到 输出长度到达了max_tokens
    FINISHED = auto()


class Sequence:
    block_size = 256
    # 全局计数器（itertools库提供），每创建一个 Sequence 对象，就从 counter 中拿一个新的整数，赋值给 seq_id
    counter = count()

    # token_ids：用户输入的 prompt 转换成 token list 中每个 token 对应的 id
    # sampling_params：采样的策略配置
    def __init__(self, token_ids: list[int], sampling_params = SamplingParams()):
        # 当前 sequence 在全局的 id
        self.seq_id = next(Sequence.counter)
        # 进入等待状态
        self.status = SequenceStatus.WAITING
        self.token_ids = copy(token_ids)
        # 最后一个 token
        self.last_token = token_ids[-1]
        # tokens 的总长度，会随着 进入decoder 阶段变长
        self.num_tokens = len(self.token_ids)
        # pormpt token 的长度
        self.num_prompt_tokens = len(token_ids)
        # 记录 有多少 token 的 KV 已经存入 cache（本质上，它是 KV cache 的进度指示器）
        # 那么 num_cached_tokens 不就等于 num_tokens - 1 吗？
        # 不完全相等，理论上生成一个 token就会缓存它的 KV，但在实际实现中因为 block、batch、流式等因素，num_cached_tokens 可能会小于 num_tokens，直到缓存完全更新。
        self.num_cached_tokens = 0
        # 存放 kv cache 的 block 的 id
        # 一个 KV cache 的 block 只存储同一个 sequence 的 KV 对应的 key 和 value。
        self.block_table = []
        # 采样的配置信息
        self.temperature = sampling_params.temperature
        self.max_tokens = sampling_params.max_tokens
        self.ignore_eos = sampling_params.ignore_eos

    def __len__(self):
        return self.num_tokens

    # 根据 索引下标 返回 tokenid
    def __getitem__(self, key):
        return self.token_ids[key]

    # 判断序列是否生成完成
    @property
    def is_finished(self):
        return self.status == SequenceStatus.FINISHED

    # 生成的新 token 数量（不含 prompt）
    @property
    def num_completion_tokens(self):
        return self.num_tokens - self.num_prompt_tokens

    # 原始用户输入的 token id 列表
    @property
    def prompt_token_ids(self):
        return self.token_ids[:self.num_prompt_tokens]

    # 已生成的 token id 列表（不含 prompt）
    @property
    def completion_token_ids(self):
        return self.token_ids[self.num_prompt_tokens:]

    # 当前 KV cache 已经存储了多少个 block
    @property
    def num_cached_blocks(self):
        return self.num_cached_tokens // self.block_size

    # 整个序列需要多少个 block 来存储
    @property
    def num_blocks(self):
        return (self.num_tokens + self.block_size - 1) // self.block_size

    # 最后一个 block 的 token 数
    @property
    def last_block_num_tokens(self):
        return self.num_tokens - (self.num_blocks - 1) * self.block_size

    # 返回第 i 个 block 的 token id
    def block(self, i):
        assert 0 <= i < self.num_blocks
        return self.token_ids[i*self.block_size: (i+1)*self.block_size]

    # 在序列末尾添加新生成 token
    def append_token(self, token_id: int):
        self.token_ids.append(token_id)
        self.last_token = token_id
        self.num_tokens += 1

    # 序列化，用于保存状态
    def __getstate__(self):
        return (self.num_tokens, self.num_prompt_tokens, self.num_cached_tokens, self.block_table,
                self.token_ids if self.num_completion_tokens == 0 else self.last_token)

    # 反序列化，用于恢复状态
    def __setstate__(self, state):
        self.num_tokens, self.num_prompt_tokens, self.num_cached_tokens, self.block_table = state[:-1]
        if self.num_completion_tokens == 0:
            self.token_ids = state[-1]
        else:
            self.last_token = state[-1]
