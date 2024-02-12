import torch
import numpy

__all__ = [
    "NumpyBitmaskTensor",
]


class NumpyBitmaskTensor:
    def __init__(self, tensor: torch.Tensor):
        self.shape = tensor.shape
        self.values, self.bitmasks, self.row_offsets = bitmask_compress(tensor.cpu())

    def decompress(self) -> torch.Tensor:
        return bitmask_decompress(self.values, self.bitmasks, self.shape)

    def to_dense(self) -> torch.Tensor:
        return self.decompress()

    @staticmethod
    def from_dense(tensor: torch.Tensor) -> "NumpyBitmaskTensor":
        return NumpyBitmaskTensor(tensor)

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
    def load(filepath: str) -> "NumpyBitmaskTensor":
        data = torch.load(filepath)
        instance = NumpyBitmaskTensor(
            torch.zeros(data["shape"])
        )  # Dummy tensor for initialization
        instance.values = data["values"].cpu()
        instance.bitmasks = data["bitmasks"].cpu()
        instance.row_offsets = data["row_offsets"].cpu()
        instance.shape = data["shape"]
        return instance

    def __repr__(self):
        return f"NumpyBitmaskTensor(shape={self.shape}, compressed=True)"


def pack_bitmasks(bitmasks: torch.Tensor) -> torch.Tensor:
    packed_bits_numpy = numpy.packbits(bitmasks.numpy(), axis=-1)
    packed_bits_torch = torch.from_numpy(packed_bits_numpy)

    return packed_bits_torch


def unpack_bitmasks(
    packed_bitmasks: torch.Tensor, original_shape: torch.Size
) -> torch.Tensor:
    # Calculate the total number of bits needed for the original shape
    total_bits_needed = numpy.prod(original_shape)

    # Unpack the bits and trim or pad the array to match the total_bits_needed
    unpacked_bits = numpy.unpackbits(
        packed_bitmasks.numpy(), axis=-1, count=original_shape[-1]
    )
    unpacked_bits_trimmed_padded = (
        unpacked_bits[:total_bits_needed]
        if unpacked_bits.size >= total_bits_needed
        else numpy.pad(
            unpacked_bits, (0, total_bits_needed - unpacked_bits.size), "constant"
        )
    )

    # Reshape to match the original shape
    unpacked_bitmasks = unpacked_bits_trimmed_padded.reshape(original_shape).astype(
        bool
    )
    unpacked_bitmasks_torch = torch.from_numpy(unpacked_bitmasks)

    return unpacked_bitmasks_torch


def bitmask_compress(tensor: torch.Tensor):
    bytemasks = tensor != 0
    row_counts = bytemasks.sum(dim=-1)
    row_offsets = torch.cumsum(row_counts, 0) - row_counts
    values = tensor[bytemasks]
    bitmasks_packed = pack_bitmasks(bytemasks)

    return values, bitmasks_packed, row_offsets


def bitmask_decompress(
    values: torch.Tensor, bitmasks: torch.Tensor, original_shape: torch.Size
) -> torch.Tensor:
    bytemasks_unpacked = unpack_bitmasks(bitmasks, original_shape)

    decompressed_tensor = torch.zeros(original_shape, dtype=values.dtype)
    decompressed_tensor[bytemasks_unpacked] = values

    return decompressed_tensor
