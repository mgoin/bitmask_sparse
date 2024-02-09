#include <cstdint>
#include <torch/extension.h>
#include <vector>

std::vector<torch::Tensor> bitmask_sparse_compression(torch::Tensor input) {
  auto options =
      torch::TensorOptions().dtype(torch::kUInt8).device(input.device());

  // Flatten input and create a mask for non-zero elements
  auto flat_input = input.flatten();
  auto nonzero_mask = flat_input != 0;

  // Extract non-zero values using masked_select
  auto values = flat_input.masked_select(nonzero_mask);

  // Prepare to create a packed bitmask
  auto bitmask_unpacked = nonzero_mask.to(torch::kUInt8);
  auto num_elements = bitmask_unpacked.size(0);
  // Ceiling division for bytes needed
  auto packed_size = (num_elements + 7) / 8; 
  auto bitmask_packed = torch::zeros({packed_size}, options);

  //
  // Simple but inefficient loop where we read/write the current byte for every element
  //
  // for (int64_t i = 0; i < num_elements; ++i) {
  //   if (bitmask_unpacked[i].item<uint8_t>()) {
  //     int64_t byte_index = i / 8;
  //     int64_t bit_index = i % 8;
  //     // Perform the operation on a retrieved scalar value
  //     auto current_value = bitmask_packed[byte_index].item<uint8_t>();
  //     current_value |= (1 << bit_index);
  //     // Assign the modified value back to the tensor
  //     bitmask_packed[byte_index] = current_value;
  //   }
  // }

  //
  // Much more effiecient but unfortunately tricky loop
  //
  uint8_t current_value = 0; // To accumulate bitmask for current byte
  int64_t last_byte_index = -1; // To track the last byte index processed

  for (int64_t i = 0; i < num_elements; ++i) {
    int64_t byte_index = i / 8;
    int64_t bit_index = i % 8;

    // Load the current byte value only when we move to a new byte
    if (byte_index != last_byte_index) {
      if (last_byte_index != -1) {
        // Store the accumulated bitmask for the previous byte
        bitmask_packed[last_byte_index] = current_value;
      }
      // Reset current value for the new byte
      current_value = bitmask_packed[byte_index].item<uint8_t>();
      last_byte_index = byte_index;
    }

    // Set the bit if the current element is non-zero
    if (bitmask_unpacked[i].item<uint8_t>()) {
      current_value |= (1 << bit_index);
    }

    // Ensure the last byte's value is written back
    if (i == num_elements - 1) {
      bitmask_packed[byte_index] = current_value;
    }
  }

  return {values, bitmask_packed};
}


torch::Tensor decompress_bitmask_sparse(torch::Tensor values,
                                        torch::Tensor bitmask_packed,
                                        std::vector<int64_t> original_shape) {
  int64_t original_size = 1;
  for (auto &dim : original_shape) {
    original_size *= dim;
  }

  auto bitmask_packed_accessor = bitmask_packed.accessor<uint8_t, 1>();
  auto decompressed_tensor = torch::zeros(original_size, values.options());

  int64_t values_idx = 0;
  for (int64_t i = 0; i < original_size; ++i) {
    int bit = i % 8;
    if ((bitmask_packed_accessor[i / 8] >> bit) & 1) {
      decompressed_tensor[i] = values[values_idx++];
    }
  }

  return decompressed_tensor.view(original_shape);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("bitmask_sparse_compression", &bitmask_sparse_compression, "Compresses a tensor into a bitmask representation and a values tensor.");
    m.def("decompress_bitmask_sparse", &decompress_bitmask_sparse, "Decompresses a tensor from the values and packed bitmask.");
}