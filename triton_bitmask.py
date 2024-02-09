import torch
import triton
import triton.language as tl
from naive_bitmask import NaiveBitmaskTensor

__all__ = [
    "TritonBitmaskTensor",
]


@triton.jit
def _or_combine(a, b):
    return a | b

@triton.jit
def triton_compute_bitmasks_and_rowcounts_kernel(
    bitmasks_ptr,
    row_counts_ptr,
    input_ptr, 
    input_row_stride, 
    bitmasks_row_stride, 
    n_cols, 
    BLOCK_SIZE: tl.constexpr):
    
    row_idx = tl.program_id(0)
    col_block_idx = tl.program_id(1)
    bitmasks_per_block = tl.cdiv(BLOCK_SIZE, 32)
    
    row_start_ptr = input_ptr + row_idx * input_row_stride 
    bitmasks_start_ptr = bitmasks_ptr + row_idx * bitmasks_row_stride + col_block_idx * bitmasks_per_block
    
    nnz = 0

    for i in tl.static_range(0, BLOCK_SIZE // 32):
        col_base_offset = col_block_idx * BLOCK_SIZE + i * 32

        col_bit_indices = tl.arange(0, 32)
        col_offsets = col_bit_indices + col_base_offset
        input_ptrs = row_start_ptr + col_offsets 

        row = tl.load(input_ptrs, mask=col_offsets < n_cols, other=0)
        row_is_nonzero = (row != 0)

        row_bits = tl.where(row_is_nonzero, 1 << col_bit_indices, 0)
        result = tl.reduce(row_bits, 0, _or_combine)
        nnz += tl.sum(row_is_nonzero)
        
        if col_base_offset < n_cols:
            tl.store(bitmasks_start_ptr + i , result)

    tl.atomic_add(row_counts_ptr + row_idx, nnz)

def triton_compute_bitmasks_and_rowcounts(x):
    x_cuda = x.cuda()
    BLOCK_SIZE=128
    
    n_rows, n_cols = x.shape
    row_counts = torch.zeros(n_rows, dtype=torch.int32, device=x_cuda.device)
    bitmasks = torch.empty(n_rows, triton.cdiv(n_cols, 32), dtype=torch.int32, device=x_cuda.device)
    triton_compute_bitmasks_and_rowcounts_kernel[(n_rows, triton.cdiv(n_cols, BLOCK_SIZE))](
        bitmasks,
        row_counts,
        x_cuda,
        x.stride(0),
        bitmasks.stride(0),
        n_cols,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    return bitmasks, row_counts


@triton.jit
def triton_bitmask_decompress_kernel(
    bitmasks_ptr,
    row_offsets_ptr,
    values_ptr,
    output_ptr,
    bitmasks_row_stride,
    n_cols):
    
    row_idx = tl.program_id(0)
    
    curr_vals_ptr = values_ptr + tl.load(row_offsets_ptr + row_idx)
    row_bitmasks_start_ptr = bitmasks_ptr + row_idx * bitmasks_row_stride 
    row_output_start_ptr = output_ptr + row_idx * n_cols

    for i in range(0, tl.cdiv(n_cols, 32)):
        col_base = i * 32
        bitmask = tl.load(row_bitmasks_start_ptr + i)
        
        col_bit_indices = tl.arange(0, 32)

        col_idx_bit_mask = (1 << col_bit_indices).to(tl.uint32, bitcast=True)
        col_low_bits_mask = col_idx_bit_mask - 1
        
        col_idx_bit_mask = col_idx_bit_mask.to(tl.int32, bitcast=True)
        col_low_bits_mask = col_low_bits_mask.to(tl.int32, bitcast=True)
        
        col_val_offsets = tl.math.popc(bitmask & col_low_bits_mask)

        values = tl.load(curr_vals_ptr + col_val_offsets)
        values = tl.where((col_idx_bit_mask & bitmask), values, 0)
        
        curr_vals_ptr += tl.math.popc(bitmask)

        col_offsets = col_base + col_bit_indices
        tl.store(row_output_start_ptr + col_offsets, values, mask=col_offsets < n_cols)

def triton_bitmask_decompress(bitmasks, row_offsets, values, n_cols):
    values_cuda = values.cuda()
    
    n_rows, _ = bitmasks.shape
    output = torch.empty(n_rows, n_cols, dtype=values.dtype, device=values_cuda.device)
    
    triton_bitmask_decompress_kernel[(n_rows, )](
        bitmasks.cuda(),
        row_offsets.cuda(),
        values_cuda,
        output,
        bitmasks.stride(0),
        n_cols)
    return output


class TritonBitmaskTensor:
    def __init__(self, tensor: torch.Tensor):
        self.device = tensor.device
        self.shape = tensor.shape
        tensor_2d = tensor.contiguous().view(-1, self.shape[-1])

        # use a triton kernel to simultaneously compute the bitmasks for each row along with 
        # with the number of non-zeros in each row
        self.bitmasks, row_counts = triton_compute_bitmasks_and_rowcounts(tensor_2d)
        # convert counts to offsets
        self.row_offsets = torch.cumsum(row_counts, 0) - row_counts
        # keep on non-zeros
        tensor_flat = tensor_2d.flatten()
        self.values = tensor_flat[tensor_flat != 0]

    def decompress(self) -> torch.Tensor:
        return triton_bitmask_decompress(self.bitmasks, self.row_offsets, self.values, self.shape[-1])\
            .view(self.shape).to(device=self.device)

    def to_dense(self) -> torch.Tensor:
        return self.decompress()

    @staticmethod
    def from_dense(tensor: torch.Tensor) -> "TritonBitmaskTensor":
        return TritonBitmaskTensor(tensor)

    def curr_memory_size_bytes(self):
        def sizeof_tensor(a):
            return a.element_size() * a.nelement()
        return sizeof_tensor(self.values) + sizeof_tensor(self.bitmasks) + sizeof_tensor(self.row_offsets)

    def save(self, filepath: str):
        torch.save(
            {
                "values": self.values,
                "bitmasks": self.bitmasks,
                "row_offsets": self.row_offsets,
                "shape": self.shape,
            },
            filepath,
        )

    @staticmethod
    def load(filepath: str) -> "TritonBitmaskTensor":
        data = torch.load(filepath)
        instance = TritonBitmaskTensor(
            torch.zeros(data["shape"])
        )  # Dummy tensor for initialization
        instance.values = data["values"]
        instance.bitmasks = data["bitmasks"]
        instance.row_offsets = data["row_offsets"]
        instance.shape = data["shape"]
        return instance

    def __repr__(self):
        return f"TritonBitmaskTensor(shape={self.shape}, compressed=True)"