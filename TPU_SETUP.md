# TPU Support for ReservoirNLP

This document provides instructions for using TPU support in the ReservoirNLP TinyStories implementation.

## Overview

The TinyStories.py file has been enhanced with comprehensive TPU support that includes:

- **Unified Device Management**: Automatic detection and initialization of TPU, GPU, or CPU
- **TPU-Optimized Data Loading**: Specialized data loader configurations for TPU performance
- **Precision Management**: Context-aware mixed precision handling for both TPU and GPU
- **Training Loop Compatibility**: TPU-specific optimizations including XLA step markers
- **Graceful Fallback**: Automatic fallback to GPU/CPU when TPU libraries are unavailable

## Quick Start

### Enable TPU Support

Set `USE_TPU = True` in the SHARED_CONFIG:

```python
SHARED_CONFIG = {
    # ... other parameters ...
    'USE_TPU': True,  # Set to True to use TPU instead of GPU
    # ... rest of parameters ...
}
```

### Fallback Behavior

If `USE_TPU = True` but torch_xla is not installed, the code will automatically:
1. Display a warning message
2. Fall back to GPU (if available) or CPU
3. Continue normal execution

## TPU Setup Instructions

### For Google Cloud TPU VMs

1. **Create a TPU VM instance:**
```bash
gcloud compute tpus tpu-vm create your-tpu-name \
  --zone=us-central1-b \
  --accelerator-type=v3-8 \
  --version=tpu-vm-pytorch-2.0
```

2. **SSH into the TPU VM:**
```bash
gcloud compute tpus tpu-vm ssh your-tpu-name --zone=us-central1-b
```

3. **Install PyTorch XLA:**
```bash
# For PyTorch 2.0+ (adjust version as needed)
pip install torch~=2.0.0 torch_xla[tpu]~=2.0.0 -f https://storage.googleapis.com/libtpu-releases/index.html
```

### For Google Colab TPU

```python
# Run this in a Colab cell before executing the main code
!pip install cloud-tpu-client==0.10 torch==2.0.0
!pip install torch_xla -f https://storage.googleapis.com/jax-releases/jax_releases.html
```

### For Other TPU Environments

Refer to the [PyTorch XLA documentation](https://pytorch.org/xla/release/2.0/index.html) for environment-specific installation instructions.

## Usage Examples

### Basic TPU Training

```python
# Set TPU configuration
SHARED_CONFIG['USE_TPU'] = True

# The code will automatically:
# 1. Initialize TPU device
# 2. Configure TPU-optimized data loaders
# 3. Use TPU-compatible training loop
# 4. Handle multiprocessing for distributed training
```

### Testing TPU Fallback

```python
# Test fallback behavior (when torch_xla not installed)
SHARED_CONFIG['USE_TPU'] = True

# Expected output:
# "Warning: TPU requested but torch_xla not installed. Falling back to GPU."
# "Using device: cuda (TPU=No)"  # or "cpu" if no GPU
```

## Key Features

### 1. Automatic Device Detection

```python
def get_device(use_tpu=False):
    """Returns appropriate device based on configuration."""
    if use_tpu:
        try:
            import torch_xla.core.xla_model as xm
            return xm.xla_device()
        except ImportError:
            print("Warning: TPU requested but torch_xla not installed. Falling back to GPU.")
            return 'cuda' if torch.cuda.is_available() else 'cpu'
    else:
        return 'cuda' if torch.cuda.is_available() else 'cpu'
```

### 2. TPU-Optimized Data Loading

- **No shuffling in DataLoader** (TPUs prefer deterministic data flow)
- **Drop last batch** for consistent batch sizes
- **No persistent workers** to avoid memory issues
- **MpDeviceLoader wrapping** for TPU compatibility

### 3. Unified Precision Handling

- **TPU**: Uses native XLA precision handling (no explicit autocast)
- **GPU**: Uses standard mixed precision with autocast
- **Dummy gradient scaler** for TPUs (XLA handles scaling internally)

### 4. TPU-Specific Training Optimizations

- **XLA step markers** (`xm.mark_step()`) for optimal TPU performance
- **Gradient reduction** (`xm.reduce_gradients()`) for distributed training
- **Slightly higher learning rates** (TPUs often benefit from this)
- **Larger batch sizes** (TPUs handle large batches efficiently)

## Performance Considerations

### TPU Optimizations Applied

1. **Batch Size**: Automatically doubled for TPUs in configuration
2. **Learning Rate**: Increased by 1.2x for TPUs
3. **Data Loading**: Disabled shuffling and persistent workers
4. **Step Marking**: Added XLA step markers for compilation efficiency

### Expected Performance

- **Faster Training**: TPUs typically provide 2-4x speedup over GPU for similar workloads
- **Better Memory Efficiency**: TPUs handle larger batch sizes effectively
- **Distributed Training**: Automatic support for multi-TPU training

## Troubleshooting

### Common Issues

1. **"torch_xla not installed"**
   - Solution: Install PyTorch XLA following setup instructions above
   - Fallback: Code will continue with GPU/CPU

2. **Memory Issues**
   - Solution: Reduce batch size in SHARED_CONFIG
   - TPU default: 2x normal batch size (automatically applied)

3. **Slow Compilation**
   - Normal: First few steps are slow due to XLA compilation
   - Optimization: Subsequent steps will be much faster

### Validation

Run the included test script to validate your setup:

```bash
python test_tpu_support.py
```

Expected output shows all tests passing and confirms TPU support is properly configured.

## Important Notes

1. **TPU VMs Required**: Standard Compute Engine instances don't support TPUs
2. **Regional Availability**: TPUs are only available in specific Google Cloud regions
3. **Cost Considerations**: TPU VMs have different pricing than regular VMs
4. **Batch Size**: TPUs work best with larger, consistent batch sizes
5. **Debugging**: Use `print()` statements for debugging (TPU debugging tools are limited)

## Migration Guide

### From GPU to TPU

1. **Change configuration**: Set `USE_TPU = True`
2. **Install dependencies**: Follow TPU setup instructions
3. **Adjust batch size**: Consider increasing for better TPU utilization
4. **Monitor performance**: First run will be slower due to compilation

### From TPU back to GPU

1. **Change configuration**: Set `USE_TPU = False`
2. **Adjust batch size**: May need to reduce for GPU memory limits
3. **No other changes needed**: Code automatically adapts

## Contributing

When modifying TPU support:

1. **Test both paths**: Always test with `USE_TPU = True` and `USE_TPU = False`
2. **Maintain fallback**: Ensure graceful fallback when torch_xla unavailable
3. **Update tests**: Add new tests to `test_tpu_support.py`
4. **Document changes**: Update this README for new features

## References

- [PyTorch XLA Documentation](https://pytorch.org/xla/)
- [Google Cloud TPU Documentation](https://cloud.google.com/tpu/docs)
- [TPU Performance Guide](https://cloud.google.com/tpu/docs/performance-guide)