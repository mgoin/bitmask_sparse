import torch
import numpy

__all__ = [
    "NumpyBitmaskTensor",
]


class NumpyBitmaskTensor:
    def __init__(self, tensor: torch.Tensor):
        self.shape = tensor.shape
        self.values, self.bitmask, self.row_offsets = bitmask_compress(tensor)

    def decompress(self) -> torch.Tensor:
        return bitmask_decompress(self.values, self.bitmask, self.shape)

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
            + sizeof_tensor(self.bitmask)
            + sizeof_tensor(self.row_offsets)
        )

    def save(self, filepath: str):
        torch.save(
            {
                "values": self.values,
                "bitmask": self.bitmask,
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
        instance.values = data["values"]
        instance.bitmask = data["bitmask"]
        instance.row_offsets = data["row_offsets"]
        instance.shape = data["shape"]
        return instance

    def __repr__(self):
        return f"NumpyBitmaskTensor(shape={self.shape}, compressed=True)"


def pack_bitmask(bitmask):
    bitmask_numpy = bitmask.numpy()
    packed_bits_numpy = numpy.packbits(bitmask_numpy)
    packed_bits_torch = torch.from_numpy(packed_bits_numpy)

    return packed_bits_torch


def unpack_bitmask(packed_bitmask, original_shape):
    # Calculate the total number of bits needed for the original shape
    total_bits_needed = numpy.prod(original_shape)

    # Unpack the bits and trim or pad the array to match the total_bits_needed
    unpacked_bits = numpy.unpackbits(packed_bitmask.numpy())
    unpacked_bits_trimmed_padded = (
        unpacked_bits[:total_bits_needed]
        if unpacked_bits.size >= total_bits_needed
        else numpy.pad(
            unpacked_bits, (0, total_bits_needed - unpacked_bits.size), "constant"
        )
    )

    # Reshape to match the original shape
    unpacked_array = unpacked_bits_trimmed_padded.reshape(original_shape)
    unpacked_torch = torch.from_numpy(unpacked_array.astype(bool))

    return unpacked_torch


def bitmask_compress(tensor):
    bitmask = tensor != 0
    row_counts = bitmask.sum(dim=-1)
    row_offsets = torch.cumsum(row_counts, 0) - row_counts
    values = tensor[bitmask]
    bitmask_packed = pack_bitmask(bitmask)

    return values, bitmask_packed, row_offsets


def bitmask_decompress(values, bitmask, original_shape):
    bitmask_unpacked = unpack_bitmask(bitmask, original_shape)

    decompressed_tensor = torch.zeros(original_shape, dtype=values.dtype)
    decompressed_tensor[bitmask_unpacked] = values

    return decompressed_tensor
