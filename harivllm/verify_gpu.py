# verify_gpu.py
import torch
import subprocess

def check_environment():
    print("🔍 Checking Environment...\n")
    
    # Check PyTorch
    print(f"PyTorch version: {torch.__version__}")
    
    # Check CUDA
    cuda_available = torch.cuda.is_available()
    print(f"CUDA available: {cuda_available}")
    
    if cuda_available:
        print(f"CUDA version: {torch.version.cuda}")
        print(f"Number of GPUs: {torch.cuda.device_count()}")
        
        for i in range(torch.cuda.device_count()):
            print(f"\nGPU {i}: {torch.cuda.get_device_name(i)}")
            print(f"  Memory: {torch.cuda.get_device_properties(i).total_memory / 1e9:.2f} GB")
            
        # Test GPU computation
        print("\n🧪 Testing GPU computation...")
        x = torch.randn(1000, 1000).cuda()
        y = torch.randn(1000, 1000).cuda()
        z = torch.matmul(x, y)
        print("✅ GPU computation successful!")
    else:
        print("❌ No GPU detected. Some modules will run slowly.")
    
    # Check NVIDIA-SMI
    try:
        result = subprocess.run(['nvidia-smi'], capture_output=True, text=True)
        print("\n📊 NVIDIA-SMI Output:")
        print(result.stdout)
    except:
        print("❌ nvidia-smi not available")

if __name__ == "__main__":
    check_environment()