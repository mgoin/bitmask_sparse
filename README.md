# bitmask_sparse

Usage:
```python
import torch
from bitmask import BitmaskTensor

dense_tensor = torch.randn([50, 50])
bitmask_tensor = BitmaskTensor.from_dense(dense_tensor)
decompressed_tensor = bitmask_tensor.to_dense()

assert torch.equal(dense_tensor, decompressed_tensor)
```

Smoke test:
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

Actual unit tests:
```bash
python -m pytest test_compression.py -v 
============================================================================= test session starts =============================================================================
platform linux -- Python 3.10.12, pytest-7.4.2, pluggy-1.3.0 -- /home/mgoin/venvs/nm/bin/python
cachedir: .pytest_cache
rootdir: /home/mgoin/code/bitmask_sparse
plugins: hydra-core-1.3.2, anyio-3.7.1
collected 12 items                                                                                                                                                            

test_compression.py::test_compress_decompress_identity[NaiveBitmaskTensor] PASSED                                                                                       [  8%]
test_compression.py::test_compress_decompress_identity[BitmaskTensor] PASSED                                                                                            [ 16%]
test_compression.py::test_compress_efficiency[NaiveBitmaskTensor] PASSED                                                                                                [ 25%]
test_compression.py::test_compress_efficiency[BitmaskTensor] PASSED                                                                                                     [ 33%]
test_compression.py::test_size_invariance[size0-0.2-NaiveBitmaskTensor] PASSED                                                                                          [ 41%]
test_compression.py::test_size_invariance[size0-0.2-BitmaskTensor] PASSED                                                                                               [ 50%]
test_compression.py::test_size_invariance[size0-0.5-NaiveBitmaskTensor] PASSED                                                                                          [ 58%]
test_compression.py::test_size_invariance[size0-0.5-BitmaskTensor] PASSED                                                                                               [ 66%]
test_compression.py::test_size_invariance[size1-0.2-NaiveBitmaskTensor] PASSED                                                                                          [ 75%]
test_compression.py::test_size_invariance[size1-0.2-BitmaskTensor] PASSED                                                                                               [ 83%]
test_compression.py::test_size_invariance[size1-0.5-NaiveBitmaskTensor] PASSED                                                                                          [ 91%]
test_compression.py::test_size_invariance[size1-0.5-BitmaskTensor] PASSED                                                                                               [100%]

============================================================================== warnings summary ===============================================================================
test_compression.py::test_compress_decompress_identity[NaiveBitmaskTensor]
test_compression.py::test_compress_decompress_identity[BitmaskTensor]
  /home/mgoin/code/bitmask_sparse/test_compression.py:18: FutureWarning: `torch.testing.assert_allclose()` is deprecated since 1.12 and will be removed in a future release. Please use `torch.testing.assert_close()` instead. You can find detailed upgrade instructions in https://github.com/pytorch/pytorch/issues/61844.
    torch.testing.assert_allclose(tensor, decompressed_tensor)

-- Docs: https://docs.pytest.org/en/stable/how-to/capture-warnings.html
======================================================================= 12 passed, 2 warnings in 1.84s ========================================================================
```
