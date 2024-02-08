import pytest
import torch

from naive_bitmask import NaiveBitmaskTensor
from bitmask import BitmaskTensor


@pytest.mark.parametrize("implementation", [NaiveBitmaskTensor, BitmaskTensor])
def test_compress_decompress_identity(implementation):
    # Create a dense tensor for testing
    tensor = torch.tensor([[1, 0, 0], [0, 2, 0], [3, 0, 4]], dtype=torch.float32)

    # Compress and then decompress the tensor
    compressed_tensor = implementation.from_dense(tensor)
    decompressed_tensor = compressed_tensor.to_dense()

    # Assert that the original and decompressed tensors are identical
    torch.testing.assert_allclose(tensor, decompressed_tensor)


@pytest.mark.parametrize("implementation", [NaiveBitmaskTensor, BitmaskTensor])
def test_compress_efficiency(implementation):
    # Create a larger, mostly sparse tensor
    tensor = torch.cat([torch.zeros(100, 100), torch.rand(100, 100)], dim=0)
    tensor[:100, :] = tensor[:100, :].apply_(lambda x: x if x > 0.95 else 0.0)

    # Compress the tensor
    compressed_tensor = implementation.from_dense(tensor)
    values = compressed_tensor.values
    bitmask_packed = compressed_tensor.bitmask_packed

    # Check that compression actually reduces size
    original_size = tensor.nelement()
    # Since bitmask_packed has 8 "indices" per byte, we should surely see compression
    compressed_size = values.nelement() + bitmask_packed.nelement()
    assert compressed_size < original_size, "Compression should reduce total size."


@pytest.mark.parametrize("implementation", [NaiveBitmaskTensor, BitmaskTensor])
@pytest.mark.parametrize("sparsity", [0.2, 0.5])
@pytest.mark.parametrize("size", [(10, 10), (100, 100)])
def test_size_invariance(implementation, sparsity, size):
    # Create a random tensor of specified size
    tensor = torch.randn(size) < sparsity

    # Compress and then decompress the tensor
    compressed_tensor = implementation.from_dense(tensor)
    decompressed_tensor = compressed_tensor.to_dense()

    # Assert that the decompressed tensor has the same shape as the original
    assert (
        decompressed_tensor.shape == tensor.shape
    ), "Decompressed tensor should retain the original shape."

    # Assert that the original and decompressed tensors are identical
    torch.testing.assert_allclose(tensor, decompressed_tensor)
