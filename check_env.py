import torch
import torch_geometric
import sys
import platform



print("="*40)
print(f"OS Platform: {platform.platform()}")
print(f"Python Ver : {sys.version.split()[0]}")
print(f"PyTorch Ver: {torch.__version__}")
print(f"CUDA Ver   : {torch.version.cuda}")
print(f"CuDNN Ver  : {torch.backends.cudnn.version()}")
print(f"PyG Ver    : {torch_geometric.__version__}")
print("-" * 40)
print(f"GPU Name   : {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")
print("="*40)