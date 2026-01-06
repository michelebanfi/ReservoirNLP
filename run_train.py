# Workaround for outdated NVIDIA drivers - must be set before torch import
import os
os.environ["PYTORCH_NVML_BASED_CUDA_CHECK"] = "0"

# ============================================================
# Architecture Selection
# ============================================================
# Set to "trm" to use the new Tiny Recursion Model architecture
# Set to "hrm" to use the original Hierarchical Reasoning Model
ARCHITECTURE = "trm"
# ============================================================

if __name__ == "__main__":
    if ARCHITECTURE == "trm":
        print("Using TRM (Tiny Recursion Model) architecture")
        from src.train_trm import train_main
    else:
        print("Using HRM (Hierarchical Reasoning Model) architecture")
        from src.train import train_main
    
    train_main()
