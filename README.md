# torch_bitmask

### Install
```
pip install git+https://github.com/mgoin/torch_bitmask
```

### Usage
```python
import torch
from torch_bitmask import NumpyBitmaskTensor as BitmaskTensor

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

bitmask_tensor: 18.0175 MB
  values: 15.9863 MB
  bitmask: 2.0000 MB
  row_offsets: 0.0312 MB
```

### Benchmark
```bash
python benchmark.py
Generating a tensor of size=[16384, 4096] and precision=torch.float32 with sparsity=0.5
Create Regular Tensor: 1.7972s
Dense memory used: 256.0000 MB

Benchmark NumpyBitmaskTensor:
  compress:   0.1037 sec
  decompress: 0.0848 sec
  memory used: 136.1031 MB

Benchmark TritonBitmaskTensor:
  compress:   0.0697 sec
  decompress: 0.0421 sec
  memory used: 136.1031 MB
```

### Tests
```bash
python -m pytest tests    
================================================ test session starts =================================================
platform linux -- Python 3.10.12, pytest-8.0.0, pluggy-1.4.0
rootdir: /home/mgoin/code/bitmask_sparse
plugins: anyio-3.7.1
collected 48 items                                                                                                   

tests/test_compression.py ................................................                                     [100%]

================================================= 48 passed in 5.61s =================================================
```