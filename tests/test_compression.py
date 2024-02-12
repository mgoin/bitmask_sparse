import pytest
import torch
import os

from torch_bitmask import (
    CppBitmaskTensor,
    NaiveBitmaskTensor,
    NumpyBitmaskTensor,
    TritonBitmaskTensor,
    Triton8BitmaskTensor,
)

IMPLS_TO_TEST = [
    CppBitmaskTensor,
    NaiveBitmaskTensor,
    NumpyBitmaskTensor,
    TritonBitmaskTensor,
    Triton8BitmaskTensor,
]

SIZES_TO_TEST = [(1, 16), (16, 1), (10, 10), (15, 15), (100, 100), (300, 300)]


@pytest.mark.parametrize("implementation", IMPLS_TO_TEST)
def test_compress_decompress_identity(implementation):
    # Create a dense tensor for testing
    tensor = torch.tensor([[1, 0, 0], [0, 2, 0], [3, 0, 4]], dtype=torch.float32)

    # Compress and then decompress the tensor
    compressed_tensor = implementation.from_dense(tensor)
    decompressed_tensor = compressed_tensor.to_dense()

    # Assert that the original and decompressed tensors are identical
    torch.testing.assert_close(tensor, decompressed_tensor)


@pytest.mark.parametrize("implementation", IMPLS_TO_TEST)
def test_compress_efficiency(implementation):
    # Create a larger, mostly sparse tensor
    tensor = torch.rand(100, 100).apply_(lambda x: x if x > 0.9 else 0.0)

    def sizeof_tensor(a):
        return a.element_size() * a.nelement()

    # Compress the tensor
    compressed_tensor = implementation.from_dense(tensor)

    # Check that compression actually reduces size
    original_size = sizeof_tensor(tensor)
    compressed_size = compressed_tensor.curr_memory_size_bytes()
    assert compressed_size < original_size / 4


@pytest.mark.parametrize("implementation", IMPLS_TO_TEST)
@pytest.mark.parametrize("sparsity", [0.5, 0.99])
@pytest.mark.parametrize("size", SIZES_TO_TEST)
def test_size_invariance(implementation, sparsity, size):
    # Create a random tensor of specified size
    tensor = torch.rand(size).apply_(lambda x: x if x > sparsity else 0.0)

    # Compress the tensor
    compressed_tensor = implementation.from_dense(tensor)

    # Assert that the original and decompressed tensors are identical
    torch.testing.assert_close(tensor, compressed_tensor.to_dense())

    # Save compressed tensor, reload, and assert they are identical
    filename = "temp.pt"
    compressed_tensor.save(filename)
    compressed_tensor = implementation.load(filename)
    torch.testing.assert_close(tensor, compressed_tensor.to_dense())
    os.remove(filename)


@pytest.mark.parametrize("implementation1", [NumpyBitmaskTensor, Triton8BitmaskTensor])
@pytest.mark.parametrize("implementation2", [NumpyBitmaskTensor, Triton8BitmaskTensor])
@pytest.mark.parametrize("sparsity", [0.5])
@pytest.mark.parametrize("size", SIZES_TO_TEST)
def test_save_load_compatibility(implementation1, implementation2, sparsity, size):
    # Create a random tensor of specified size
    tensor = torch.rand(size).apply_(lambda x: x if x > sparsity else 0.0)

    # Compress the tensor
    compressed1_tensor = implementation1.from_dense(tensor)
    compressed2_tensor = implementation2.from_dense(tensor)

    # Assert that the original and decompressed tensors are identical
    torch.testing.assert_close(tensor, compressed1_tensor.to_dense())
    torch.testing.assert_close(tensor, compressed2_tensor.to_dense())

    # Save compressed1 tensor, load as compressed2 tensor, and assert they are identical
    filename = "temp.pt"
    compressed1_tensor.save(filename)
    compressed1_tensor_as_2 = implementation2.load(filename)
    torch.testing.assert_close(tensor, compressed1_tensor_as_2.to_dense())
    os.remove(filename)
