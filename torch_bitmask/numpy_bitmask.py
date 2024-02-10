import torch
import numpy

__all__ = [
    "NumpyBitmaskTensor",
]


class NumpyBitmaskTensor:
    def __init__(self, tensor: torch.Tensor):
        self.shape = tensor.shape
        self.bitmask_packed, self.values = bitmask_compress(tensor)

    def decompress(self) -> torch.Tensor:
        return bitmask_decompress(self.bitmask_packed, self.values, self.shape)

    def to_dense(self) -> torch.Tensor:
        return self.decompress()

    @staticmethod
    def from_dense(tensor: torch.Tensor) -> "NumpyBitmaskTensor":
        return NumpyBitmaskTensor(tensor)

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
    def load(filepath: str) -> "NumpyBitmaskTensor":
        data = torch.load(filepath)
        instance = NumpyBitmaskTensor(
            torch.zeros(data["shape"])
        )  # Dummy tensor for initialization
        instance.values = data["values"]
        instance.bitmask_packed = data["bitmask_packed"]
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
    values = tensor[bitmask]
    bitmask_packed = pack_bitmask(bitmask)

    return bitmask_packed, values


def bitmask_decompress(bitmask, values, original_shape):
    bitmask_unpacked = unpack_bitmask(bitmask, original_shape)

    decompressed_tensor = torch.zeros(original_shape, dtype=values.dtype)
    decompressed_tensor[bitmask_unpacked] = values

    return decompressed_tensor
