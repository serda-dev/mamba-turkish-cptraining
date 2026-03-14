"""Environment and dependency checks."""

import logging
import sys
from typing import Dict, Any

logger = logging.getLogger(__name__)


def check_environment(verbose: bool = True) -> Dict[str, Any]:
    """
    Check and print environment information.
    
    Returns dict with version info and status.
    """
    info = {}
    
    # Python
    info["python_version"] = sys.version.split()[0]
    
    # PyTorch
    try:
        import torch
        info["torch_version"] = torch.__version__
        info["torch_cuda_version"] = torch.version.cuda
        info["cuda_available"] = torch.cuda.is_available()
        
        if torch.cuda.is_available():
            info["cuda_device_count"] = torch.cuda.device_count()
            info["cuda_device_name"] = torch.cuda.get_device_name(0)
            info["cuda_memory_total_gb"] = torch.cuda.get_device_properties(0).total_memory / 1e9
            info["cuda_memory_allocated_gb"] = torch.cuda.memory_allocated() / 1e9
    except ImportError:
        info["torch_version"] = "NOT INSTALLED"
        info["cuda_available"] = False
    
    # Transformers
    try:
        import transformers
        info["transformers_version"] = transformers.__version__
    except ImportError:
        info["transformers_version"] = "NOT INSTALLED"
    
    # Jamba-related kernel packages
    try:
        import mamba_ssm
        info["mamba_ssm_version"] = mamba_ssm.__version__
    except ImportError:
        info["mamba_ssm_version"] = "NOT INSTALLED"
    except AttributeError:
        info["mamba_ssm_version"] = "installed (no __version__)"
    
    try:
        import causal_conv1d
        info["causal_conv1d_version"] = getattr(causal_conv1d, "__version__", "installed")
    except ImportError:
        info["causal_conv1d_version"] = "NOT INSTALLED"

    try:
        import bitsandbytes as bnb
        info["bitsandbytes_version"] = getattr(bnb, "__version__", "installed")
    except ImportError:
        info["bitsandbytes_version"] = "NOT INSTALLED"
    
    # Einops
    try:
        import einops
        info["einops_version"] = einops.__version__
    except ImportError:
        info["einops_version"] = "NOT INSTALLED"
    
    if verbose:
        print("=" * 60)
        print("Environment Check")
        print("=" * 60)
        print(f"Python:        {info['python_version']}")
        print(f"PyTorch:       {info.get('torch_version', 'N/A')}")
        print(f"CUDA (torch):  {info.get('torch_cuda_version', 'N/A')}")
        print(f"CUDA avail:    {info.get('cuda_available', False)}")
        
        if info.get("cuda_available"):
            print(f"GPU:           {info.get('cuda_device_name', 'N/A')}")
            print(f"GPU Memory:    {info.get('cuda_memory_total_gb', 0):.1f} GB")
        
        print(f"Transformers:  {info.get('transformers_version', 'N/A')}")
        print(f"mamba-ssm:     {info.get('mamba_ssm_version', 'N/A')}")
        print(f"causal_conv1d: {info.get('causal_conv1d_version', 'N/A')}")
        print(f"bitsandbytes:  {info.get('bitsandbytes_version', 'N/A')}")
        print(f"einops:        {info.get('einops_version', 'N/A')}")
        print("=" * 60)
        
        # Warnings
        if not info.get("cuda_available"):
            print("⚠️  WARNING: CUDA not available! Training will be slow on CPU.")
        
        if info.get("mamba_ssm_version") == "NOT INSTALLED":
            print("⚠️  WARNING: mamba-ssm not installed. Jamba fast kernels will be unavailable.")
        if info.get("bitsandbytes_version") == "NOT INSTALLED":
            print("⚠️  WARNING: bitsandbytes not installed. adamw_8bit optimizer will not work.")
    
    return info


def quick_gpu_test() -> bool:
    """Quick test that GPU is accessible and working."""
    try:
        import torch
        if not torch.cuda.is_available():
            return False
        
        # Quick tensor operation
        x = torch.randn(10, 10, device="cuda")
        y = x @ x.T
        del x, y
        torch.cuda.empty_cache()
        
        return True
    except Exception as e:
        logger.error(f"GPU test failed: {e}")
        return False


if __name__ == "__main__":
    check_environment()
    
    print("\nRunning quick GPU test...")
    if quick_gpu_test():
        print("✓ GPU test passed")
    else:
        print("✗ GPU test failed")
