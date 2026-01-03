# Workaround for outdated NVIDIA drivers - must be set before torch import
import os
os.environ["PYTORCH_NVML_BASED_CUDA_CHECK"] = "0"

from src.train import train_main

if __name__ == "__main__":
    train_main()
