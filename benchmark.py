import time
import timeit
import torch

from torch_bitmask import (
    NaiveBitmaskTensor,
    TritonBitmaskTensor,
    CppBitmaskTensor,
    NumpyBitmaskTensor,
)

tensor_impls = [NumpyBitmaskTensor, TritonBitmaskTensor]
shape = [16 * 1024, 4 * 1024]
dtype = torch.float32
sparsity = 0.5

dense_tensor = torch.rand(shape).to(dtype)
print(
    f"Generating a tensor of size={shape} and precision={dtype} with sparsity={sparsity}"
)
mask = (dense_tensor.abs() < (1 - sparsity)).int()
dense_tensor = dense_tensor * mask


def create_regular_tensor():
    _ = torch.rand(shape).to(dtype)


def compress_tensor_impl(input_tensor, tensor_impl):
    return tensor_impl.from_dense(input_tensor)


def decompress_tensor_impl(compressed_tensor):
    return compressed_tensor.to_dense()


def benchmark_implementation(input_tensor, tensor_impl, iters=10):
    # Warmup
    compressed_tensor = compress_tensor_impl(input_tensor, tensor_impl)
    compressed_size_bytes = compressed_tensor.curr_memory_size_bytes()
    decompress_tensor_impl(compressed_tensor)

    # Measure compression time
    start = time.perf_counter()
    for i in range(iters):
        compress_tensor_impl(input_tensor, tensor_impl)
    compress_duration = (time.perf_counter() - start) / iters

    # Measure decompression time
    start = time.perf_counter()
    for i in range(iters):
        decompress_tensor_impl(compressed_tensor)
    decompress_duration = (time.perf_counter() - start) / iters

    return compress_duration, decompress_duration, compressed_size_bytes


# Benchmarking
print(f"Create Regular Tensor: {timeit.timeit(create_regular_tensor, number=10):.4f}s")
print(
    f"Dense memory used: {dense_tensor.element_size() * dense_tensor.nelement() / 1024**2:.4f} MB"
)
print()

for impl in tensor_impls:
    print(f"Benchmark {impl.__name__}:")
    (
        compress_duration,
        decompress_duration,
        compressed_size_bytes,
    ) = benchmark_implementation(dense_tensor, impl)
    print(f"  compress:   {compress_duration:.4f} sec")
    print(f"  decompress: {decompress_duration:.4f} sec")
    print(f"  memory used: {compressed_size_bytes / 1024**2:.4f} MB")
    print()
