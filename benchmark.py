import timeit
import torch

from naive_bitmask import NaiveBitmaskTensor
from triton_bitmask import TritonBitmaskTensor
from bitmask import BitmaskTensor

SHAPE = [1024, 1024]


def create_naive_bitmask_tensor():
    dense_tensor = torch.randn(SHAPE)
    _ = NaiveBitmaskTensor.from_dense(dense_tensor)


def decompress_naive_bitmask_tensor():
    dense_tensor = torch.randn(SHAPE)
    naive_bitmask_tensor = NaiveBitmaskTensor.from_dense(dense_tensor)
    _ = naive_bitmask_tensor.to_dense()


def create_regular_tensor():
    _ = torch.randn(SHAPE)


def create_bitmask_tensor():
    dense_tensor = torch.randn(SHAPE)
    _ = BitmaskTensor.from_dense(dense_tensor)


def decompress_bitmask_tensor():
    dense_tensor = torch.randn(SHAPE)
    bitmask_tensor = BitmaskTensor.from_dense(dense_tensor)
    _ = bitmask_tensor.to_dense()


def create_trition_bitmask_tensor():
    dense_tensor = torch.randn(SHAPE)
    _ = TritonBitmaskTensor.from_dense(dense_tensor)


def decompress_trition_bitmask_tensor():
    dense_tensor = torch.randn(SHAPE)
    bitmask_tensor = TritonBitmaskTensor.from_dense(dense_tensor)
    _ = bitmask_tensor.to_dense()


# Benchmarking
print(f"SHAPE = {SHAPE}")
# print(
#     "Create NaiveBitmaskTensor:", timeit.timeit(create_naive_bitmask_tensor, number=10)
# )
# print(
#     "Decompress NaiveBitmaskTensor:",
#     timeit.timeit(decompress_naive_bitmask_tensor, number=10),
# )
print("Create Regular Tensor:", timeit.timeit(create_regular_tensor, number=10))
print("Create BitmaskTensor:", timeit.timeit(create_bitmask_tensor, number=10))
print("Decompress BitmaskTensor:", timeit.timeit(decompress_bitmask_tensor, number=10))
print(
    "Create TritonBitmaskTensor:",
    timeit.timeit(create_trition_bitmask_tensor, number=10),
)
print(
    "Decompress TritonBitmaskTensor:",
    timeit.timeit(decompress_trition_bitmask_tensor, number=10),
)
