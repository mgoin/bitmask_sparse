import torch
import triton
import triton.language as tl

__all__ = [
    "Triton8BitmaskTensor",
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
    BLOCK_SIZE: tl.constexpr,
):
    # Compute the row index and the block index for columns based on the program's ID.
    row_idx = tl.program_id(0)
    col_block_idx = tl.program_id(1)
    bitmasks_per_block = tl.cdiv(BLOCK_SIZE, 8)

    # Calculate starting pointers for the current row in both the input and bitmask tensors.
    row_start_ptr = input_ptr + row_idx * input_row_stride
    bitmasks_start_ptr = (
        bitmasks_ptr
        + row_idx * bitmasks_row_stride
        + col_block_idx * bitmasks_per_block
    )

    # Initialize the count of non-zero elements found in this row.
    nnz = 0

    # Iterate over each group of 8 columns within the block since we are using int32 for bitmask element.
    for i in tl.static_range(0, BLOCK_SIZE // 8):
        col_base_offset = col_block_idx * BLOCK_SIZE + i * 8

        # Calculate pointers to the input elements for this chunk.
        col_bit_indices = tl.arange(0, 8)
        col_offsets = col_bit_indices + col_base_offset
        input_ptrs = row_start_ptr + col_offsets

        # Load elements from the input tensor, applying a mask for bounds checking.
        row = tl.load(input_ptrs, mask=col_offsets < n_cols, other=0)
        row_is_nonzero = row != 0

        # For each non-zero element, set the corresponding bit in a 8-bit integer.
        row_bits = tl.where(row_is_nonzero, 1 << col_bit_indices, 0)
        # Combine the bits into a single 8-bit integer.
        result = tl.reduce(row_bits, 0, _or_combine)
        # Accumulate the count of non-zero elements.
        nnz += tl.sum(row_is_nonzero)

        # Store the resulting 8-bit integer into the bitmask tensor if within bounds.
        if col_base_offset < n_cols:
            tl.store(bitmasks_start_ptr + i, result)

    # Atomically add the count of non-zero elements found in this row to the row_counts tensor.
    tl.atomic_add(row_counts_ptr + row_idx, nnz)


def triton_compute_bitmasks_and_rowcounts(x):
    x_cuda = x.cuda()
    BLOCK_SIZE = 128
    assert BLOCK_SIZE % 8 == 0

    n_rows, n_cols = x.shape
    row_counts = torch.zeros(n_rows, dtype=torch.int32, device=x_cuda.device)
    bitmasks = torch.empty(
        n_rows, triton.cdiv(n_cols, 8), dtype=torch.uint8, device=x_cuda.device
    )
    triton_compute_bitmasks_and_rowcounts_kernel[
        (n_rows, triton.cdiv(n_cols, BLOCK_SIZE))
    ](
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
    bitmasks_ptr, row_offsets_ptr, values_ptr, output_ptr, bitmasks_row_stride, n_cols
):
    # Obtain the current row index based on the program ID.
    row_idx = tl.program_id(0)

    # Calculate the pointers to the current row's values, bitmask data, and output tensor using the row offset.
    curr_vals_ptr = values_ptr + tl.load(row_offsets_ptr + row_idx)
    row_bitmasks_start_ptr = bitmasks_ptr + row_idx * bitmasks_row_stride
    row_output_start_ptr = output_ptr + row_idx * n_cols

    # Iterate through each 8-bit segment of the bitmask for the current row.
    for i in range(0, tl.cdiv(n_cols, 8)):
        col_base = i * 8
        # Load the 8-bit bitmask segment.
        bitmask = tl.load(row_bitmasks_start_ptr + i)

        # Generate a sequence of 8 indices (0 to 7) for the bitmask.
        col_bit_indices = tl.arange(0, 8)

        # Generate masks for extracting values based on the bitmask.
        col_idx_bit_mask = (1 << col_bit_indices).to(tl.uint32, bitcast=True)
        col_low_bits_mask = col_idx_bit_mask - 1

        # Convert masks to int32 for compatibility.
        col_idx_bit_mask = col_idx_bit_mask.to(tl.int32, bitcast=True)
        col_low_bits_mask = col_low_bits_mask.to(tl.int32, bitcast=True)
        expanded_bitmask = bitmask.to(tl.uint32).to(tl.int32, bitcast=True)

        num_non_zeros = tl.math.popc(expanded_bitmask)

        if num_non_zeros > 0:
            # Calculate the offset for each value based on the number of set bits (popcount) in the bitmask up to that point.
            col_val_offsets = tl.math.popc(bitmask & col_low_bits_mask)

            # Load the values corresponding to non-zero bits in the bitmask.
            values = tl.load(curr_vals_ptr + col_val_offsets)
            # Use the bitmask to conditionally select values or zeros.
            values = tl.where((col_idx_bit_mask & bitmask), values, 0)
        else:
            values = tl.zeros((8,), dtype=values_ptr.dtype.element_ty)

        # Increment the pointer for the next set of values based on the number of non-zero elements in this segment.
        curr_vals_ptr += num_non_zeros

        # Calculate the column offsets for this segment and store the values in the output tensor.
        col_offsets = col_base + col_bit_indices
        tl.store(row_output_start_ptr + col_offsets, values, mask=col_offsets < n_cols)


def triton_bitmask_decompress(bitmasks, row_offsets, values, shape):
    values_cuda = values.cuda()

    n_rows, n_cols = shape
    output = torch.empty(n_rows, n_cols, dtype=values.dtype, device=values_cuda.device)

    triton_bitmask_decompress_kernel[(n_rows,)](
        bitmasks.cuda(),
        row_offsets.cuda(),
        values_cuda,
        output,
        bitmasks.stride(0),
        n_cols,
    )
    return output


class Triton8BitmaskTensor:
    def __init__(self, tensor: torch.Tensor):
        self.device = tensor.device
        self.shape = tensor.shape
        tensor_2d = tensor.contiguous().view(-1, self.shape[-1]).cuda()

        # Use a triton kernel to simultaneously compute the bitmasks for each row along with
        # with the number of non-zeros in each row
        self.bitmasks, row_counts = triton_compute_bitmasks_and_rowcounts(tensor_2d)

        # Convert counts to offsets
        self.row_offsets = torch.cumsum(row_counts, 0) - row_counts

        # Keep only the non-zero values
        tensor_flat = tensor_2d.flatten()
        self.values = tensor_flat[tensor_flat != 0]

    def decompress(self) -> torch.Tensor:
        return (
            triton_bitmask_decompress(
                self.bitmasks, self.row_offsets, self.values, self.shape
            )
            .view(self.shape)
            .to(device=self.device)
        )

    def to_dense(self) -> torch.Tensor:
        return self.decompress()

    @staticmethod
    def from_dense(tensor: torch.Tensor) -> "Triton8BitmaskTensor":
        return Triton8BitmaskTensor(tensor)

    def curr_memory_size_bytes(self):
        def sizeof_tensor(a):
            return a.element_size() * a.nelement()

        return (
            sizeof_tensor(self.values)
            + sizeof_tensor(self.bitmasks)
            + sizeof_tensor(self.row_offsets)
        )

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
    def load(filepath: str) -> "Triton8BitmaskTensor":
        data = torch.load(filepath)
        instance = Triton8BitmaskTensor(
            torch.zeros(data["shape"])
        )  # Dummy tensor for initialization
        instance.values = data["values"]
        instance.bitmasks = data["bitmasks"]
        instance.row_offsets = data["row_offsets"]
        instance.shape = data["shape"]
        return instance

    def __repr__(self):
        return f"Triton8BitmaskTensor(shape={self.shape}, compressed=True)"
