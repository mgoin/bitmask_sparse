import pytest
import torch
import os

from torch_bitmask import (
    NaiveBitmaskTensor,
    BitmaskTensor,
    TritonBitmaskTensor,
    NumpyBitmaskTensor,
)

IMPLS_TO_TEST = [
    NaiveBitmaskTensor,
    BitmaskTensor,
    TritonBitmaskTensor,
    NumpyBitmaskTensor,
]


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
    tensor = torch.cat([torch.zeros(100, 100), torch.rand(100, 100)], dim=0)
    tensor[:100, :] = tensor[:100, :].apply_(lambda x: x if x > 0.95 else 0.0)

    def sizeof_tensor(a):
        return a.element_size() * a.nelement()

    # Compress the tensor
    compressed_tensor = implementation.from_dense(tensor)

    # Check that compression actually reduces size
    original_size = sizeof_tensor(tensor)
    compressed_size = compressed_tensor.curr_memory_size_bytes()
    assert compressed_size < original_size, "Compression should reduce total size."


@pytest.mark.parametrize("implementation", IMPLS_TO_TEST)
@pytest.mark.parametrize("sparsity", [0.2, 0.5])
@pytest.mark.parametrize("size", [(1, 16), (10, 10), (15, 15), (100, 100), (300, 300)])
def test_size_invariance(implementation, sparsity, size):
    # Create a random tensor of specified size
    tensor = torch.randn(size) < sparsity

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
