from collections import deque

import numpy as np
import xxhash

from nanovllm.engine.sequence import Sequence


class Block:

    def __init__(self, block_id):
        self.block_id = block_id
        self.ref_count = 0
        self.hash = -1
        self.token_ids = []

    def update(self, hash: int, token_ids: list[int]):
        self.hash = hash
        self.token_ids = token_ids

    def reset(self):
        self.ref_count = 1
        self.hash = -1
        self.token_ids = []


class BlockManager:

    def __init__(self, num_blocks: int, block_size: int):
        assert num_blocks > 0
        self.block_size = block_size
        self.blocks: list[Block] = [Block(i) for i in range(num_blocks)]
        self.hash_to_block_id: dict[int, int] = dict()
        self.free_block_ids: deque[int] = deque(range(num_blocks))
        self.used_block_ids: set[int] = set()

    @classmethod
    def compute_hash(cls, token_ids: list[int], prefix: int = -1):
        h = xxhash.xxh64()
        if prefix != -1:
            h.update(prefix.to_bytes(8, "little"))
        h.update(np.array(token_ids).tobytes())
        return h.intdigest()

    def _allocate_block(self, block_id: int) -> Block:
        block = self.blocks[block_id]
        assert block.ref_count == 0
        block.reset()
        self.free_block_ids.remove(block_id)
        self.used_block_ids.add(block_id)
        return self.blocks[block_id]

    # def _deallocate_block(self, block_id: int) -> Block:
    #     assert self.blocks[block_id].ref_count == 0
    #     self.used_block_ids.remove(block_id)
    #     # 重置block状态，清除hash值，避免状态污染
    #     self.blocks[block_id].reset()
    #     self.blocks[block_id].ref_count = 0  # reset()会设置ref_count=1，需要改回0
    #     self.free_block_ids.append(block_id)
    def _deallocate_block(self, block_id: int) -> Block:
        '''
        释放一个block，前提是该block的引用计数为0
        1. 确保该block当前未被使用（ref_count为0）
        2. 从已使用block集合中移除该block_id
        3. 从hash映射中移除该block（如果有hash值）
        4. 重置block状态（hash和token_ids）
        5. 将该block_id添加到空闲block列表中
        '''
        block = self.blocks[block_id]
        assert block.ref_count == 0
        self.used_block_ids.remove(block_id)
        
        # 从hash映射中移除该block（如果有hash值）
        if block.hash != -1:
            # 移除hash映射，避免已释放的block仍然被缓存命中
            if self.hash_to_block_id.get(block.hash) == block_id:
                del self.hash_to_block_id[block.hash]
        
        # 重置block状态，清除hash值，避免状态污染
        block.reset()
        block.ref_count = 0  # reset()会设置ref_count=1，需要改回0
        self.free_block_ids.append(block_id)

    def can_allocate(self, seq: Sequence) -> bool:
        return len(self.free_block_ids) >= seq.num_blocks

    def allocate(self, seq: Sequence):
        assert not seq.block_table
        h = -1
        cache_miss = False
        for i in range(seq.num_blocks):
            token_ids = seq.block(i)
            h = (
                self.compute_hash(token_ids, h)
                if len(token_ids) == self.block_size
                else -1
            )
            block_id = self.hash_to_block_id.get(h, -1)
            if block_id == -1 or self.blocks[block_id].token_ids != token_ids:
                cache_miss = True
            if cache_miss:
                block_id = self.free_block_ids[0]
                block = self._allocate_block(block_id)
            else:
                seq.num_cached_tokens += self.block_size
                if block_id in self.used_block_ids:
                    block = self.blocks[block_id]
                    block.ref_count += 1
                else:
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
            # 需要分配一个新的block；若前一block尚未写入hash（被抢占重新分配等场景），补写hash
            # if last_block.hash == -1:
            #     token_ids = seq.block(seq.num_blocks-1)
            #     prefix = self.blocks[block_table[-2]].hash if len(block_table) > 1 else -1
            #     h = self.compute_hash(token_ids, prefix)
            #     last_block.update(h, token_ids)
            #     self.hash_to_block_id[h] = last_block.block_id
            block_id = self.free_block_ids[0]
            self._allocate_block(block_id)
            block_table.append(block_id)
        elif len(seq) % self.block_size == 0:
        #     assert last_block.hash == -1
        #     token_ids = seq.block(seq.num_blocks - 1)
        #     prefix = self.blocks[block_table[-2]].hash if len(block_table) > 1 else -1
        #     h = self.compute_hash(token_ids, prefix)
        #     last_block.update(h, token_ids)
        #     self.hash_to_block_id[h] = last_block.block_id
        # else:
        #     assert last_block.hash == -1
            # 当前block刚好满了，需要更新hash值（如果还没有设置）
            if last_block.hash == -1:
                # 只有当hash还没有设置时才计算和更新
                token_ids = seq.block(seq.num_blocks-1)
                prefix = self.blocks[block_table[-2]].hash if len(block_table) > 1 else -1
                h = self.compute_hash(token_ids, prefix)
                last_block.update(h, token_ids)
                self.hash_to_block_id[h] = last_block.block_id
            # 如果 last_block.hash != -1，说明hash已经在allocate阶段设置了，无需重复设置
        else:
            # 当前block还有空间，hash应该还没有设置
            # 但是如果这个block是从allocate阶段复用的，hash可能已经设置了
            # 所以不需要断言 hash == -1
            pass
