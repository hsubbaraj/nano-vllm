#!/usr/bin/env python3
"""
GPU Environment Verification Script
Checks if the development environment is properly configured
"""

import torch
import subprocess
import sys

def check_environment():
    print("🔍 Checking Environment...\n")
    
    # Check Python version
    print(f"Python version: {sys.version}")
    print(f"Python executable: {sys.executable}\n")
    
    # Check PyTorch
    print(f"PyTorch version: {torch.__version__}")
    
    # Check CUDA
    cuda_available = torch.cuda.is_available()
    print(f"CUDA available: {cuda_available}")
    
    if cuda_available:
        print(f"CUDA version: {torch.version.cuda}")
        print(f"cuDNN version: {torch.backends.cudnn.version()}")
        print(f"Number of GPUs: {torch.cuda.device_count()}")
        
        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            print(f"\nGPU {i}: {torch.cuda.get_device_name(i)}")
            print(f"  Compute Capability: {props.major}.{props.minor}")
            print(f"  Memory: {props.total_memory / 1e9:.2f} GB")
            print(f"  Multiprocessors: {props.multi_processor_count}")
            
        # Test GPU computation
        print("\n🧪 Testing GPU computation...")
        try:
            # Simple matrix multiplication
            x = torch.randn(1000, 1000).cuda()
            y = torch.randn(1000, 1000).cuda()
            z = torch.matmul(x, y)
            torch.cuda.synchronize()
            print("✅ GPU computation successful!")
            
            # Test memory allocation
            print("\n💾 Testing memory allocation...")
            try:
                large_tensor = torch.zeros(10000, 10000).cuda()
                print(f"✅ Successfully allocated {large_tensor.element_size() * large_tensor.nelement() / 1e9:.2f} GB")
                del large_tensor
                torch.cuda.empty_cache()
            except RuntimeError as e:
                print(f"⚠️  Large allocation failed: {e}")
                
        except Exception as e:
            print(f"❌ GPU computation failed: {e}")
    else:
        print("❌ No GPU detected. Some modules will run slowly.")
        print("\nPossible reasons:")
        print("  - NVIDIA drivers not installed")
        print("  - Docker not configured for GPU access")
        print("  - Running on a CPU-only machine")
    
    # Check NVIDIA-SMI
    print("\n" + "="*50)
    try:
        result = subprocess.run(['nvidia-smi', '--query-gpu=name,memory.total,memory.free,utilization.gpu', '--format=csv,noheader'], 
                              capture_output=True, text=True)
        if result.returncode == 0:
            print("📊 NVIDIA-SMI GPU Status:")
            print(result.stdout)
        else:
            print("❌ nvidia-smi failed")
    except FileNotFoundError:
        print("❌ nvidia-smi not found in PATH")
    except Exception as e:
        print(f"❌ Error running nvidia-smi: {e}")
    
    # Check key libraries
    print("\n📦 Checking key libraries:")
    libraries = [
        ('transformers', 'import transformers; transformers.__version__'),
        ('accelerate', 'import accelerate; accelerate.__version__'),
        ('triton', 'import triton; triton.__version__'),
        ('flash_attn', 'import flash_attn; flash_attn.__version__'),
    ]
    
    for lib_name, version_cmd in libraries:
        try:
            version = eval(version_cmd)
            print(f"  ✅ {lib_name}: {version}")
        except ImportError:
            print(f"  ❌ {lib_name}: Not installed")
        except Exception as e:
            print(f"  ⚠️  {lib_name}: Error - {e}")
    
    # Summary
    print("\n" + "="*50)
    if cuda_available:
        print("✅ Environment is ready for GPU-accelerated inference!")
    else:
        print("⚠️  Environment is set up but GPU acceleration is not available.")
        print("   You can still run the tutorials but performance will be limited.")

if __name__ == "__main__":
    check_environment()