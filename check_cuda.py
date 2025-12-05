#!/usr/bin/env python3
"""
CUDA Diagnostics Script
Helps diagnose why CUDA might not be available for PyTorch
"""

import sys

def check_cuda():
    """Check CUDA availability and print diagnostic information"""
    print("=" * 70)
    print("CUDA Diagnostics")
    print("=" * 70)

    # Check PyTorch
    try:
        import torch
        print(f"✓ PyTorch installed: {torch.__version__}")
    except ImportError:
        print("✗ PyTorch not installed")
        print("  Install with: pip install torch torchvision")
        return False

    # Check CUDA availability
    cuda_available = torch.cuda.is_available()
    print(f"\nCUDA available: {'✓ YES' if cuda_available else '✗ NO'}")

    if cuda_available:
        print(f"\nCUDA Details:")
        print(f"  CUDA version: {torch.version.cuda}")
        print(f"  cuDNN version: {torch.backends.cudnn.version()}")
        print(f"  Number of GPUs: {torch.cuda.device_count()}")

        for i in range(torch.cuda.device_count()):
            print(f"\n  GPU {i}:")
            print(f"    Name: {torch.cuda.get_device_name(i)}")
            print(f"    Memory: {torch.cuda.get_device_properties(i).total_memory / 1024**3:.1f} GB")
            print(f"    Compute capability: {torch.cuda.get_device_properties(i).major}.{torch.cuda.get_device_properties(i).minor}")

        # Test GPU
        print(f"\nTesting GPU computation...")
        try:
            x = torch.randn(1000, 1000).cuda()
            y = torch.randn(1000, 1000).cuda()
            z = torch.matmul(x, y)
            print("  ✓ GPU computation successful")
        except Exception as e:
            print(f"  ✗ GPU computation failed: {e}")

        return True
    else:
        print("\nPossible issues:")
        print("\n1. CPU-only PyTorch installed")
        print("   Check with: python -c \"import torch; print(torch.__version__)\"")
        print("   Current version:", torch.__version__)

        if '+cpu' in torch.__version__ or 'cpu' in torch.__version__:
            print("   → This is a CPU-only build!")
            print("   Solution: Reinstall PyTorch with CUDA support")
            print("   Visit: https://pytorch.org/get-started/locally/")

        print("\n2. CUDA toolkit not installed")
        try:
            import subprocess
            result = subprocess.run(['nvcc', '--version'], capture_output=True, text=True)
            if result.returncode == 0:
                print("   ✓ CUDA toolkit found:")
                for line in result.stdout.split('\n'):
                    if 'release' in line.lower():
                        print(f"     {line.strip()}")
            else:
                print("   ✗ nvcc not found")
        except FileNotFoundError:
            print("   ✗ CUDA toolkit not found (nvcc not in PATH)")
            print("   Download from: https://developer.nvidia.com/cuda-downloads")

        print("\n3. NVIDIA drivers")
        try:
            result = subprocess.run(['nvidia-smi'], capture_output=True, text=True)
            if result.returncode == 0:
                print("   ✓ NVIDIA drivers found:")
                for line in result.stdout.split('\n')[:2]:
                    if line.strip():
                        print(f"     {line}")
            else:
                print("   ✗ nvidia-smi failed")
        except FileNotFoundError:
            print("   ✗ nvidia-smi not found")
            print("   Install NVIDIA drivers from: https://www.nvidia.com/Download/index.aspx")

        print("\n" + "=" * 70)
        print("Quick Fix:")
        print("=" * 70)
        print("\nMost likely cause: PyTorch CPU-only version installed")
        print("\nTo fix, reinstall PyTorch with CUDA support:")
        print("\n  # For CUDA 11.8")
        print("  pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu118")
        print("\n  # For CUDA 12.1")
        print("  pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu121")
        print("\nCheck your CUDA version with: nvcc --version")
        print("Or visit: https://pytorch.org/get-started/locally/")
        print("=" * 70)

        return False

if __name__ == "__main__":
    success = check_cuda()
    sys.exit(0 if success else 1)
