import torch


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


def bitmask_sparse_compression(tensor):
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

    return values, bitmask_unpacked, bitmask_packed


# Example usage
x = torch.tensor([[1, 0, 2], [5, 3, 0], [4, 0, 5]], dtype=torch.float32)
values, bitmask_unpacked, bitmask_packed = bitmask_sparse_compression(x)

print("Values:", values)
print("Bitmask Unpacked:", bitmask_unpacked)
print(
    "Bitmask Packed Binary:",
    ["{0:b}".format(b.item()).zfill(8) for b in bitmask_packed],
)

from torch.utils.cpp_extension import load

bitmask_lib = load(
    name="bitmask_lib",
    sources=["bitmask_sparse_extension.cpp"],
    verbose=True,
)

values, bitmask_packed = bitmask_lib.bitmask_sparse_compression(x)
print("Values:", values)
print(
    "Bitmask Packed Binary:",
    ["{0:b}".format(b.item()).zfill(8) for b in bitmask_packed],
)

# Assume you have the 'values' and 'bitmask_packed' tensors from the previous compression step
values = torch.tensor([1.0, 2.0, 5.0, 3.0, 4.0, 5.0], dtype=torch.float32)
bitmask_packed = torch.tensor([93, 1], dtype=torch.uint8)  # '01011101', '00000001'
original_shape = [3, 3]

# Decompress the tensor
decompressed_tensor = bitmask_lib.decompress_bitmask_sparse(
    values, bitmask_packed, original_shape
)
print("Decompressed Tensor:\n", decompressed_tensor)

assert torch.allclose(x, decompressed_tensor)
