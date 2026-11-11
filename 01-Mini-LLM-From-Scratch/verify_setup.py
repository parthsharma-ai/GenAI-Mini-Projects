import torch
import numpy as np

print("PyTorch version:", torch.__version__)
print("NumPy version:", np.__version__)

x = torch.tensor([1, 2, 3])

print("Test tensor:", x)
print("CUDA available:", torch.cuda.is_available())

print("\nEnvironment setup successful!")