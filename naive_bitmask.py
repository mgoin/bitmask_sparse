import torch

__all__ = [
    "NaiveBitmaskTensor",
]


class NaiveBitmaskTensor:
    def __init__(self, tensor: torch.Tensor):
        self.shape = tensor.shape
        self.values, self.bitmask_packed = naive_bitmask_sparse_compression(tensor)

    def decompress(self) -> torch.Tensor:
        return naive_decompress_bitmask_sparse(
            self.values, self.bitmask_packed, self.shape
        )

    def to_dense(self) -> torch.Tensor:
        return self.decompress()

    @staticmethod
    def from_dense(tensor: torch.Tensor) -> "NaiveBitmaskTensor":
        return NaiveBitmaskTensor(tensor)

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
    def load(filepath: str) -> "NaiveBitmaskTensor":
        data = torch.load(filepath)
        instance = NaiveBitmaskTensor(
            torch.zeros(data["shape"])
        )  # Dummy tensor for initialization
        instance.values = data["values"]
        instance.bitmask_packed = data["bitmask_packed"]
        instance.shape = data["shape"]
        return instance

    def __repr__(self):
        return f"NaiveBitmaskTensor(shape={self.shape}, compressed=True)"


def pack_bits_to_byte(bitmask):
    """
    Pack a tensor of bits into bytes by treating each 8 bits as one byte.
    Args:
    bitmask (torch.Tensor): The tensor of bits (0 or 1 values).

    Returns:
    torch.Tensor: A new tensor where each 8 bits are packed into a single byte.
    """
    # Ensure the bitmask is flat
    bitmask = bitmask.flatten()
    # Calculate the necessary padding to make the length a multiple of 8
    padding_size = (-len(bitmask)) % 8
    # Pad the bitmask with zeros at the end if necessary
    if padding_size > 0:
        bitmask = torch.cat(
            [bitmask, torch.zeros(padding_size, dtype=torch.uint8)], dim=0
        )

    # Reshape the bitmask to have a size that can be divided by 8
    reshaped = bitmask.view(-1, 8)
    # Convert groups of 8 bits into bytes
    packed = torch.zeros(len(reshaped), dtype=torch.uint8)
    for i in range(8):
        packed |= (reshaped[:, i] << i).to(torch.uint8)
    return packed


def naive_bitmask_sparse_compression(tensor: torch.Tensor) -> tuple:
    """
    Compresses a tensor into a packed bitmask representation and a values tensor.

    Args:
    tensor (torch.Tensor): The input tensor to compress.

    Returns:
    tuple of torch.Tensor: A tuple containing the values tensor and the packed bitmask tensor.
    """
    # Flatten the tensor to simplify the extraction process
    flat_tensor = tensor.flatten()

    # Extract non-zero values
    values = flat_tensor[flat_tensor != 0]

    # Create a bitmask where each bit represents whether the corresponding tensor element is non-zero
    bitmask_unpacked = (flat_tensor != 0).to(torch.uint8)

    # Pack the bitmask to store 8 elements per byte
    bitmask_packed = pack_bits_to_byte(bitmask_unpacked)

    return values, bitmask_packed


def unpack_bitmask_to_bits(
    bitmask_packed: torch.Tensor, original_size: int
) -> torch.Tensor:
    """
    Unpacks a packed byte tensor into a bit tensor.

    Args:
    bitmask_packed (torch.Tensor): The packed byte tensor.
    original_size (int): The total number of bits (elements) in the original uncompressed bitmask.

    Returns:
    torch.Tensor: The unpacked bitmask as bits.
    """
    # Calculate the number of bytes and unpack each byte into 8 bits
    unpacked_bits = torch.zeros(original_size, dtype=torch.uint8)
    for i, byte in enumerate(bitmask_packed):
        for bit in range(8):
            if i * 8 + bit < original_size:
                unpacked_bits[i * 8 + bit] = (byte >> bit) & 1
    return unpacked_bits


def naive_decompress_bitmask_sparse(
    values: torch.Tensor, bitmask_packed: torch.Tensor, original_shape: tuple
) -> torch.Tensor:
    """
    Decompresses a tensor from the values and packed bitmask.

    Args:
    values (torch.Tensor): The non-zero values tensor.
    bitmask_packed (torch.Tensor): The packed bitmask tensor.
    original_shape (tuple): The shape of the original uncompressed tensor.

    Returns:
    torch.Tensor: The decompressed tensor.
    """
    original_size = torch.prod(torch.tensor(original_shape)).item()
    bitmask_unpacked = unpack_bitmask_to_bits(bitmask_packed, original_size)

    # Initialize a tensor for the decompressed output
    decompressed_tensor = torch.zeros(original_size, dtype=values.dtype)

    # Place the values back according to the bitmask
    values_idx = 0
    for i, bit in enumerate(bitmask_unpacked):
        if bit == 1:
            decompressed_tensor[i] = values[values_idx]
            values_idx += 1

    # Reshape the decompressed tensor to its original shape
    decompressed_tensor = decompressed_tensor.view(original_shape)

    return decompressed_tensor
