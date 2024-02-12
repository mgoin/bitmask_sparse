from setuptools import setup

setup(
    name="torch_bitmask",
    version="0.0.1",
    author="Michael Goin",
    author_email="mgoin64@gmail.com",
    description="Implementation of compressed sparse tensors for PyTorch",
    install_requires=["torch>2", "numpy", "ninja", "triton"],
    packages=["torch_bitmask"],
)
