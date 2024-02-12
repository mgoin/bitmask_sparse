from importlib import import_module

PACKAGE_NAME = "torch_bitmask"


# Only when the class is accessed will the __getattr__ trigger the actual import
# of the module and class. This controls when the import side-effects happen,
# such as compilation, effectively delaying them until the class is actually needed.
class LazyLoader:
    def __init__(self, module_relative_path, class_name):
        self.module_relative_path = module_relative_path
        self.class_name = class_name
        self.module = None

    def __getattr__(self, name):
        if self.module is None:
            # Dynamically import the module and class when first accessed
            module = import_module(self.module_relative_path, PACKAGE_NAME)
            self.module = getattr(module, self.class_name)
        return getattr(self.module, name)


# Define lazy-loaded classes
CppBitmaskTensor = LazyLoader(".cpp_bitmask", "CppBitmaskTensor")
NaiveBitmaskTensor = LazyLoader(".naive_bitmask", "NaiveBitmaskTensor")
NumpyBitmaskTensor = LazyLoader(".numpy_bitmask", "NumpyBitmaskTensor")
TritonBitmaskTensor = LazyLoader(".triton_bitmask", "TritonBitmaskTensor")
Triton8BitmaskTensor = LazyLoader(".triton_8bitmask", "Triton8BitmaskTensor")

__all__ = [
    "CppBitmaskTensor",
    "NaiveBitmaskTensor",
    "NumpyBitmaskTensor",
    "TritonBitmaskTensor",
    "Triton8BitmaskTensor",
]
