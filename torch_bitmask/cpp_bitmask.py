import torch
import os

from torch.utils.cpp_extension import load

SCRIPT_DIR = os.path.dirname(os.path.realpath(__file__))

bitmask_lib = load(
    name="bitmask_lib",
    sources=[f"{SCRIPT_DIR}/bitmask_sparse_extension.cpp"],
    extra_cflags=["-O2"],
)


__all__ = [
    "CppBitmaskTensor",
]


class CppBitmaskTensor:
    def __init__(self, tensor: torch.Tensor):
        self.shape = tensor.shape
        self.values, self.bitmask_packed = bitmask_sparse_compression(tensor)

    def decompress(self) -> torch.Tensor:
        return decompress_bitmask_sparse(self.values, self.bitmask_packed, self.shape)

    def to_dense(self) -> torch.Tensor:
        return self.decompress()

    @staticmethod
    def from_dense(tensor: torch.Tensor) -> "CppBitmaskTensor":
        return CppBitmaskTensor(tensor)

    def curr_memory_size_bytes(self):
        def sizeof_tensor(a):
            return a.element_size() * a.nelement()

        return sizeof_tensor(self.values) + sizeof_tensor(self.bitmask_packed)

    def save(self, filepath: str):
        torch.save(
            {
                "values": self.values,
                "bitmask_packed": self.bitmask_packed,
                "shape": self.shape,
            },
            filepath,
        )

    @staticmethod
    def load(filepath: str) -> "CppBitmaskTensor":
        data = torch.load(filepath)
        instance = CppBitmaskTensor(
            torch.zeros(data["shape"])
        )  # Dummy tensor for initialization
        instance.values = data["values"]
        instance.bitmask_packed = data["bitmask_packed"]
        instance.shape = data["shape"]
        return instance

    def __repr__(self):
        return f"CppBitmaskTensor(shape={self.shape}, compressed=True)"


def bitmask_sparse_compression(tensor: torch.Tensor) -> tuple:
    return bitmask_lib.bitmask_sparse_compression(tensor)


def decompress_bitmask_sparse(
    values: torch.Tensor, bitmask_packed: torch.Tensor, original_shape: tuple
) -> torch.Tensor:
    return bitmask_lib.decompress_bitmask_sparse(values, bitmask_packed, original_shape)
