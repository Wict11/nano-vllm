from collections import deque # 双端队列，用于高效的块分配和释放
import xxhash # 高速哈希函数库
import numpy as np # 数值计算库，主要用于处理 token ID 列表

from nanovllm.engine.sequence import Sequence # 导入序列类

'''
    实现块分配、释放和缓存机制，支持高效的 KV 缓存复用。
    通过哈希计算检测重复块，减少重复计算和内存浪费。
    主要用于处理变长序列的注意力计算。
'''

class Block:

    def __init__(self, block_id):
        self.block_id = block_id
        self.ref_count = 0 # 引用计数，表示有多少序列在使用该块，为0时表示块为空闲
        self.hash = -1 # 块内容的哈希值，用于缓存匹配，-1表示未设置
        self.token_ids = []

    def update(self, hash: int, token_ids: list[int]):
        self.hash = hash # 更新块的哈希值
        self.token_ids = token_ids # 更新块的 token ID 列表

    def reset(self):
        self.ref_count = 1 # 重置引用计数为1，表示刚分配给一个序列
        self.hash = -1 # 重置哈希值
        self.token_ids = [] # 清空 token ID 列表


class BlockManager:
    # 总块数 和 块大小
    def __init__(self, num_blocks: int, block_size: int):
        self.block_size = block_size
        # 创建总块数个 Block 对象
        self.blocks: list[Block] = [Block(i) for i in range(num_blocks)]
        # 哈希值到块 ID 的映射字典
        self.hash_to_block_id: dict[int, int] = dict()
        # 空闲块 ID 双端队列
        self.free_block_ids: deque[int] = deque(range(num_blocks))
        # 已用块 ID 集合
        self.used_block_ids: set[int] = set()

    @classmethod #类方法
    # 使用 xxhash 计算 token IDs 的哈希值。
    # 如果有前缀，则先更新前缀哈希，再更新 token IDs。
    # 返回 64 位整数哈希，用于检测块内容是否相同。
    def compute_hash(cls, token_ids: list[int], prefix: int = -1):
        h = xxhash.xxh64()
        # 累计前缀哈希值
        if prefix != -1:
            h.update(prefix.to_bytes(8, "little"))
        h.update(np.array(token_ids).tobytes())
        return h.intdigest()

    # 分配指定块 ID 的块
    def _allocate_block(self, block_id: int) -> Block:
        block = self.blocks[block_id]
        assert block.ref_count == 0
        block.reset()
        # 更新空闲和已用块 ID 集合
        self.free_block_ids.remove(block_id)
        self.used_block_ids.add(block_id)
        return self.blocks[block_id]

    # 释放指定块 ID 的块
    def _deallocate_block(self, block_id: int) -> Block:
        assert self.blocks[block_id].ref_count == 0
        self.used_block_ids.remove(block_id)
        self.free_block_ids.append(block_id)

    # 检查是否可以为序列分配足够的块
    def can_allocate(self, seq: Sequence) -> bool:
        # 序列需要的块数 小于等于 空闲块数
        return len(self.free_block_ids) >= seq.num_blocks

    
    def allocate(self, seq: Sequence):
        '''
            为序列分配块，更新块表和缓存
            在prefill阶段执行，对于一个sequence只执行一次
        '''
        assert not seq.block_table
        h = -1
        cache_miss = False
        # 遍历序列的每个块
        for i in range(seq.num_blocks):
            token_ids = seq.block(i) # 这里的token ids指的是一个块内保存的 token id 范围
            # 计算块的哈希值，检查缓存
            # 只对完整的块计算哈希，因为只有稳定的块才可被复用。最后一个块可能不完整
            # 原理是同一个【token id 列表】对应的【哈希值】是唯一的
            h = self.compute_hash(token_ids, h) if len(token_ids) == self.block_size else -1
            # 如果找得到缓存块id就返回对应的块id，如果找不到就返回 -1
            block_id = self.hash_to_block_id.get(h, -1)
            # 如果块未命中缓存，或者内容不匹配，则标记为缓存未命中
            if block_id == -1 or self.blocks[block_id].token_ids != token_ids:
                cache_miss = True
            if cache_miss:
                # 出现的原因是没有在_deallocate_block的时候删除hash映射
                # 缓存未命中，分配新块
                block_id = self.free_block_ids[0]
                block = self._allocate_block(block_id)
            else:
                # 缓存命中且内容匹配则复用已有块
                seq.num_cached_tokens += self.block_size
                if block_id in self.used_block_ids:
                    block = self.blocks[block_id]
                    block.ref_count += 1
                else:
                    block = self._allocate_block(block_id)
            if h != -1:
                # 更新块内容和哈希映射
                block.update(h, token_ids)
                self.hash_to_block_id[h] = block_id
            # 填充块表
            seq.block_table.append(block_id)


    def deallocate(self, seq: Sequence):
        # 逆序遍历
        for block_id in reversed(seq.block_table):
            block = self.blocks[block_id]
            block.ref_count -= 1
            if block.ref_count == 0:
                self._deallocate_block(block_id)
        seq.num_cached_tokens = 0
        seq.block_table.clear()

    # 检查是否可以为序列追加块
    def can_append(self, seq: Sequence) -> bool:
        # 在decode阶段被调用，只有decode到了新块开始时才需要分配新块
        return len(self.free_block_ids) >= (len(seq) % self.block_size == 1)

    # 追加新生成的 token 到序列对应的块中
    # 其实是在decode已经解码到下一个token时才会对上一个已经满了block进行哈希处理，并分配新块
    def may_append(self, seq: Sequence):
        block_table = seq.block_table
        last_block = self.blocks[block_table[-1]]
        # 如果追加后新块开始，则分配新块
        if len(seq) % self.block_size == 1:
            assert last_block.hash != -1
            block_id = self.free_block_ids[0]
            self._allocate_block(block_id)
            block_table.append(block_id)
        # 如果最后一个块已满，则更新其哈希值
        # 已满的块才有稳定的哈希值，可以用于缓存
        elif len(seq) % self.block_size == 0:
            assert last_block.hash == -1
            token_ids = seq.block(seq.num_blocks-1)
            # 取出上一块的哈希值作为前缀
            prefix = self.blocks[block_table[-2]].hash if len(block_table) > 1 else -1
            h = self.compute_hash(token_ids, prefix)
            last_block.update(h, token_ids)
            self.hash_to_block_id[h] = last_block.block_id
        else:
            assert last_block.hash == -1
