# Workaround for outdated NVIDIA drivers - must be set before torch import
import os
os.environ["PYTORCH_NVML_BASED_CUDA_CHECK"] = "0"
os.environ["PYTORCH_NO_CUDA_MEMORY_CACHING"] = "1"  # Disable caching allocator to avoid NVML calls
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Import torch and configure backends BEFORE importing our package
import torch
if torch.cuda.is_available():
    torch.backends.cuda.enable_flash_sdp(False)
    torch.backends.cuda.enable_mem_efficient_sdp(False)
    torch.backends.cuda.enable_math_sdp(True)

if __name__ == "__main__":
    print("Using Pure Reasoning architecture (no T5)")
    from pure_reasoning.train import train_main
    train_main()
