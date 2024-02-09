# bitmask_sparse

### Usage
```python
import torch
from bitmask import BitmaskTensor

dense_tensor = torch.randn([50, 50])
bitmask_tensor = BitmaskTensor.from_dense(dense_tensor)
decompressed_tensor = bitmask_tensor.to_dense()

assert torch.equal(dense_tensor, decompressed_tensor)
```

### Compression Demo
```bash
python demo.py
Generating a tensor of size=[4096, 4096] and precision=torch.float16 with sparsity=0.5

dense_tensor: 32.0000 MB

bitmask_tensor: 18.0229 MB
  values: 15.9916 MB
  bitmasks: 2.0000 MB
  row_offsets: 0.0312 MB
```

### Benchmark
```bash
python benchmark.py
SHAPE = [1024, 1024]
Create Regular Tensor: 0.040321263999999246
Create BitmaskTensor: 5.361965241
Decompress BitmaskTensor: 15.432242420999998
Create TritonBitmaskTensor: 0.5133457880000023
Decompress TritonBitmaskTensor: 0.20679421799999886
```

### Tests
```bash
python -m pytest test_compression.py -v 
================================================== test session starts ===================================================
platform linux -- Python 3.10.12, pytest-8.0.0, pluggy-1.4.0 -- /home/mgoin/venvs/nm/bin/python
cachedir: .pytest_cache
rootdir: /home/mgoin/code/bitmask_sparse
plugins: anyio-3.7.1
collected 36 items                                                                                                       

test_compression.py::test_compress_decompress_identity[NaiveBitmaskTensor] PASSED                                  [  2%]
test_compression.py::test_compress_decompress_identity[BitmaskTensor] PASSED                                       [  5%]
test_compression.py::test_compress_decompress_identity[TritonBitmaskTensor] PASSED                                 [  8%]
test_compression.py::test_compress_efficiency[NaiveBitmaskTensor] PASSED                                           [ 11%]
test_compression.py::test_compress_efficiency[BitmaskTensor] PASSED                                                [ 13%]
test_compression.py::test_compress_efficiency[TritonBitmaskTensor] PASSED                                          [ 16%]
test_compression.py::test_size_invariance[size0-0.2-NaiveBitmaskTensor] PASSED                                     [ 19%]
test_compression.py::test_size_invariance[size0-0.2-BitmaskTensor] PASSED                                          [ 22%]
test_compression.py::test_size_invariance[size0-0.2-TritonBitmaskTensor] PASSED                                    [ 25%]
test_compression.py::test_size_invariance[size0-0.5-NaiveBitmaskTensor] PASSED                                     [ 27%]
test_compression.py::test_size_invariance[size0-0.5-BitmaskTensor] PASSED                                          [ 30%]
test_compression.py::test_size_invariance[size0-0.5-TritonBitmaskTensor] PASSED                                    [ 33%]
test_compression.py::test_size_invariance[size1-0.2-NaiveBitmaskTensor] PASSED                                     [ 36%]
test_compression.py::test_size_invariance[size1-0.2-BitmaskTensor] PASSED                                          [ 38%]
test_compression.py::test_size_invariance[size1-0.2-TritonBitmaskTensor] PASSED                                    [ 41%]
test_compression.py::test_size_invariance[size1-0.5-NaiveBitmaskTensor] PASSED                                     [ 44%]
test_compression.py::test_size_invariance[size1-0.5-BitmaskTensor] PASSED                                          [ 47%]
test_compression.py::test_size_invariance[size1-0.5-TritonBitmaskTensor] PASSED                                    [ 50%]
test_compression.py::test_size_invariance[size2-0.2-NaiveBitmaskTensor] PASSED                                     [ 52%]
test_compression.py::test_size_invariance[size2-0.2-BitmaskTensor] PASSED                                          [ 55%]
test_compression.py::test_size_invariance[size2-0.2-TritonBitmaskTensor] PASSED                                    [ 58%]
test_compression.py::test_size_invariance[size2-0.5-NaiveBitmaskTensor] PASSED                                     [ 61%]
test_compression.py::test_size_invariance[size2-0.5-BitmaskTensor] PASSED                                          [ 63%]
test_compression.py::test_size_invariance[size2-0.5-TritonBitmaskTensor] PASSED                                    [ 66%]
test_compression.py::test_size_invariance[size3-0.2-NaiveBitmaskTensor] PASSED                                     [ 69%]
test_compression.py::test_size_invariance[size3-0.2-BitmaskTensor] PASSED                                          [ 72%]
test_compression.py::test_size_invariance[size3-0.2-TritonBitmaskTensor] PASSED                                    [ 75%]
test_compression.py::test_size_invariance[size3-0.5-NaiveBitmaskTensor] PASSED                                     [ 77%]
test_compression.py::test_size_invariance[size3-0.5-BitmaskTensor] PASSED                                          [ 80%]
test_compression.py::test_size_invariance[size3-0.5-TritonBitmaskTensor] PASSED                                    [ 83%]
test_compression.py::test_size_invariance[size4-0.2-NaiveBitmaskTensor] PASSED                                     [ 86%]
test_compression.py::test_size_invariance[size4-0.2-BitmaskTensor] PASSED                                          [ 88%]
test_compression.py::test_size_invariance[size4-0.2-TritonBitmaskTensor] PASSED                                    [ 91%]
test_compression.py::test_size_invariance[size4-0.5-NaiveBitmaskTensor] PASSED                                     [ 94%]
test_compression.py::test_size_invariance[size4-0.5-BitmaskTensor] PASSED                                          [ 97%]
test_compression.py::test_size_invariance[size4-0.5-TritonBitmaskTensor] PASSED                                    [100%]

=================================================== 36 passed in 6.29s ===================================================
```

### Smoke test
```bash
python smoke_test.py
Values: tensor([1., 2., 5., 3., 4., 5.])
Bitmask Unpacked: tensor([1, 0, 1, 1, 1, 0, 1, 0, 1], dtype=torch.uint8)
Bitmask Packed Binary: ['01011101', '00000001']
Using /home/mgoin/.cache/torch_extensions/py310_cu121 as PyTorch extensions root...
Emitting ninja build file /home/mgoin/.cache/torch_extensions/py310_cu121/bitmask_lib/build.ninja...
Building extension module bitmask_lib...
Allowing ninja to set a default number of workers... (overridable by setting the environment variable MAX_JOBS=N)
ninja: no work to do.
Loading extension module bitmask_lib...
Values: tensor([1., 2., 5., 3., 4., 5.])
Bitmask Packed Binary: ['01011101', '00000001']
Decompressed Tensor:
 tensor([[1., 0., 2.],
        [5., 3., 0.],
        [4., 0., 5.]])
```
