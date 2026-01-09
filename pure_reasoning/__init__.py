# Pure Reasoning Package
# Note: Do NOT import torch here - it triggers CUDA initialization
# The NVML workaround must be set in the runner script before any torch import
import os
os.environ.setdefault("PYTORCH_NVML_BASED_CUDA_CHECK", "0")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
