import torch

class KVCacheManager:

    def __init__(self, num_blocks, block_size, num_layers, num_heads, head_dim, device):
        self.num_blocks = num_blocks
        self.block_size = block_size
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.device = device

        self.key_blocks = [
            torch.zeros((num_blocks, block_size, num_heads, head_dim), dtype=torch.float16, device=device)
            for _ in range(num_layers)
        ]

        self.value_blocks = [
            torch.zeros((num_blocks, block_size, num_heads, head_dim), dtype=torch.float16, device=device)
            for _ in range(num_layers)
        ]

        self.free_block_ids = list(range(num_blocks))
        self.block_usage = {}

    def allocate_blocks(self, request_id, num_blocks_needed):
        if len(self.free_block_ids) < num_blocks_needed:
            raise ValueError("Not enough free KV cache block available!")
        allocated = [self.free_block_ids.pop(0) for _ in range(num_blocks_needed)]
        self.block_usage[request_id] = allocated
        return allocated

    def free_blocks(self, request_id):
        if request_id not in self.block_usage:
            return
        for block_id in self.block_usage[request_id]:
            self.free_block_ids.append(block_id)
        del self.block_usage[request_id]

    def get_block_indices(self, request_id):
        return self.block_usage.get(request_id, [])

    def get_kv_blocks(self, block_indices, layer):
        key = self.key_blocks[layer][block_indices]
        value = self.value_blocks[layer][block_indices]
        return key, value

    def write_to_blocks(self, layer, block_indices, token_offset, key_tensor, value_tensor):
        for i, block_id in enumerate(block_indices):
            self.key_blocks[layer][block_id, token_offset] = key_tensor[i]
            self.value_blocks[layer][block_id, token_offset] = value_tensor[i]

    def num_free_blocks(self):
        return len(self.free_block_ids)