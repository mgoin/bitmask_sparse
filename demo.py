import torch
from torch_bitmask import NumpyBitmaskTensor as BitmaskTensor

shape = [4096, 4096]
dtype = torch.float16
dense_tensor = torch.rand(shape, dtype=dtype)

sparsity = 0.5
print(
    f"Generating a tensor of size={shape} and precision={dtype} with sparsity={sparsity}\n"
)
mask = (dense_tensor.abs() < (1 - sparsity)).int()
sparse_tensor = dense_tensor * mask

bitmask_tensor = BitmaskTensor.from_dense(sparse_tensor)


def sizeof_tensor(a):
    return a.element_size() * a.nelement()


print(f"dense_tensor: {sizeof_tensor(dense_tensor) / 1024**2:.4f} MB\n")
print(f"bitmask_tensor: {bitmask_tensor.curr_memory_size_bytes() / 1024**2:.4f} MB")
print(f"  values: {sizeof_tensor(bitmask_tensor.values) / 1024**2:.4f} MB")
print(f"  bitmasks: {sizeof_tensor(bitmask_tensor.bitmasks) / 1024**2:.4f} MB")
print(f"  row_offsets: {sizeof_tensor(bitmask_tensor.row_offsets) / 1024**2:.4f} MB")

assert torch.equal(sparse_tensor, bitmask_tensor.to_dense())
