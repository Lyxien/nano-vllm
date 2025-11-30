from collections import deque
import xxhash
import numpy as np

from nanovllm.engine.sequence import Sequence

# TODO：多个 sequence 可以共享 1个block
class Block:

    def __init__(self, block_id):
        # Block 的唯一编号
        self.block_id = block_id
        # 当前有多少 sequence 正在使用这个 block
        # 例如，两个 sequence 的起始语句都是 How are you：
        #   How are you，Li?
        #   How are you，Fan?
        self.ref_count = 0
        # 对 block 计算 hash 值，可以通过 hash 快速判断两个 block 是否相等
        # block 能够命中 = token_ids.size() 个 token 和 当前要处理的 token 完全一致
        self.hash = -1
        # 当前 block 对应 token id
        self.token_ids = []

    def update(self, hash: int, token_ids: list[int]):
        self.hash = hash
        self.token_ids = token_ids

    def reset(self):
        self.ref_count = 1
        self.hash = -1
        self.token_ids = []


# 构造全局唯一对象，目的是管理 Block
class BlockManager:

    # Block的数量 和 每个Block的大小
    def __init__(self, num_blocks: int, block_size: int):
        assert num_blocks > 0
        self.block_size = block_size
        # 用 list 存放所有的 block
        self.blocks: list[Block] = [Block(i) for i in range(num_blocks)]
        # 哈希值 → BlockId 的映射（实现复用，可以判断 Block 是否重复）
        # 只有 block 放满我们才会对它进行 hash 计算
        self.hash_to_block_id: dict[int, int] = dict()
        # 空闲 block pool（队列里只存放 block_id）
        self.free_block_ids: deque[int] = deque(range(num_blocks))
        # 已分配给某些 sequence 的 blockId
        self.used_block_ids: set[int] = set()

    @classmethod
    def compute_hash(cls, token_ids: list[int], prefix: int = -1):
        h = xxhash.xxh64()
        if prefix != -1:
            h.update(prefix.to_bytes(8, "little"))
        h.update(np.array(token_ids).tobytes())
        return h.intdigest()

    # 新分配 block，并初始化（ref_count=1）
    def _allocate_block(self, block_id: int) -> Block:
        block = self.blocks[block_id]
        assert block.ref_count == 0
        block.reset()
        self.free_block_ids.remove(block_id)
        self.used_block_ids.add(block_id)
        return self.blocks[block_id]

    # 释放 block。（ref_count 变成 0 时才能释放。）    
    def _deallocate_block(self, block_id: int) -> Block:
        assert self.blocks[block_id].ref_count == 0
        self.used_block_ids.remove(block_id)
        self.free_block_ids.append(block_id)

    # prefill 阶段：是否能为 seq 分配足够 block
    def can_allocate(self, seq: Sequence) -> bool:
        return len(self.free_block_ids) >= seq.num_blocks

    # allocate(seq)，prefill 阶段的核心逻辑
    def allocate(self, seq: Sequence):
        assert not seq.block_table
        h = -1
        cache_miss = False
        # 遍历 Sequence 的 blocks
        # TODO：Sequence 的 blocks 是从哪里来的  
        for i in range(seq.num_blocks):
            # 返回第 i 个 block 的所有 token id
            token_ids = seq.block(i)
            # 计算当前 token id 列表的 hash 值（如果block不是满的话则返回-1）
            h = self.compute_hash(token_ids, h) if len(token_ids) == self.block_size else -1
            # 根据 hash 值找到对应的块
            block_id = self.hash_to_block_id.get(h, -1)
            # 如果 cache没找到 或者 token_ids 不完全一致 -> 未命中 Cache
            if block_id == -1 or self.blocks[block_id].token_ids != token_ids:
                cache_miss = True
            if cache_miss:
                # 如果未命中，从 空闲队列中 取出 1个block 分配
                block_id = self.free_block_ids[0]
                block = self._allocate_block(block_id)
            else:
                # 如果命中了
                # 当前 sequence 被 cache 的 token 数量 += block_size 
                seq.num_cached_tokens += self.block_size
                if block_id in self.used_block_ids:
                    # 检查这个 block 是否已经存在，如果 block 已经存在 → 可以复用（出现 hash 命中）
                    block = self.blocks[block_id]
                    block.ref_count += 1
                else:
                    # 如果 block 不存在 -> 这个block没有被缓存过 -> 新建一个 block
                    # TODO：这里为什么会不存在呢？不是已经根据 hash 找到了 block 吗？
                    block = self._allocate_block(block_id)
            if h != -1:
                block.update(h, token_ids)
                self.hash_to_block_id[h] = block_id
            seq.block_table.append(block_id)

    def deallocate(self, seq: Sequence):
        for block_id in reversed(seq.block_table):
            block = self.blocks[block_id]
            block.ref_count -= 1
            if block.ref_count == 0:
                self._deallocate_block(block_id)
        seq.num_cached_tokens = 0
        seq.block_table.clear()

    def can_append(self, seq: Sequence) -> bool:
        return len(self.free_block_ids) >= (len(seq) % self.block_size == 1)

    def may_append(self, seq: Sequence):
        block_table = seq.block_table
        last_block = self.blocks[block_table[-1]]
        if len(seq) % self.block_size == 1:
            assert last_block.hash != -1
            block_id = self.free_block_ids[0]
            self._allocate_block(block_id)
            block_table.append(block_id)
        elif len(seq) % self.block_size == 0:
            assert last_block.hash == -1
            token_ids = seq.block(seq.num_blocks-1)
            prefix = self.blocks[block_table[-2]].hash if len(block_table) > 1 else -1
            h = self.compute_hash(token_ids, prefix)
            last_block.update(h, token_ids)
            self.hash_to_block_id[h] = last_block.block_id
        else:
            assert last_block.hash == -1
