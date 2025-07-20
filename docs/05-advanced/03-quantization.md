# Quantization: Model Compression for Efficient Inference

## Overview

Quantization reduces the precision of model weights and activations, enabling faster inference and lower memory usage with minimal accuracy loss. This guide covers various quantization techniques, from basic INT8 quantization to advanced methods like GPTQ and AWQ.

## Fundamentals of Quantization

### What is Quantization?

Quantization maps continuous values to a discrete set:
- **FP16**: 16-bit floating point (standard for inference)
- **INT8**: 8-bit integer (2x memory reduction)
- **INT4**: 4-bit integer (4x memory reduction)
- **Binary/Ternary**: 1-2 bits (extreme compression)

```python
# Basic quantization concept
def quantize(tensor, n_bits=8):
    qmin = -(2**(n_bits-1))
    qmax = 2**(n_bits-1) - 1
    
    scale = (tensor.max() - tensor.min()) / (qmax - qmin)
    zero_point = qmin - tensor.min() / scale
    
    quantized = torch.round(tensor / scale + zero_point)
    quantized = torch.clamp(quantized, qmin, qmax)
    
    return quantized.to(torch.int8), scale, zero_point

def dequantize(quantized, scale, zero_point):
    return (quantized.float() - zero_point) * scale
```

### Quantization Schemes

1. **Symmetric Quantization**: Zero point is 0
2. **Asymmetric Quantization**: Arbitrary zero point
3. **Per-Tensor**: Single scale for entire tensor
4. **Per-Channel**: Different scale per channel/row
5. **Dynamic**: Compute scale at runtime
6. **Static**: Pre-computed scale

## INT8 Quantization

### Basic INT8 Implementation

```python
import torch
import torch.nn as nn
from typing import Tuple, Optional

class INT8Linear(nn.Module):
    """INT8 quantized linear layer."""
    
    def __init__(self, in_features: int, out_features: int, bias: bool = True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        
        # Quantized weights
        self.register_buffer('weight_int8', torch.zeros(out_features, in_features, dtype=torch.int8))
        self.register_buffer('weight_scale', torch.zeros(out_features, 1))
        self.register_buffer('weight_zero_point', torch.zeros(out_features, 1, dtype=torch.int8))
        
        if bias:
            self.register_buffer('bias', torch.zeros(out_features))
        else:
            self.register_buffer('bias', None)
        
        # Input quantization parameters
        self.input_scale = None
        self.input_zero_point = None
        
    def quantize_weights(self, weight: torch.Tensor):
        """Quantize FP weights to INT8."""
        # Per-channel quantization for better accuracy
        w_min = weight.min(dim=1, keepdim=True)[0]
        w_max = weight.max(dim=1, keepdim=True)[0]
        
        # Calculate scale and zero point
        scale = (w_max - w_min) / 255.0
        zero_point = (-w_min / scale).round()
        
        # Quantize
        weight_int8 = ((weight - w_min) / scale).round().clamp(0, 255).to(torch.int8)
        
        self.weight_int8 = weight_int8 - zero_point.to(torch.int8)
        self.weight_scale = scale
        self.weight_zero_point = zero_point.to(torch.int8)
    
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        # Dynamic input quantization
        if self.input_scale is None:
            inp_min = input.min()
            inp_max = input.max()
            self.input_scale = (inp_max - inp_min) / 255.0
            self.input_zero_point = (-inp_min / self.input_scale).round().to(torch.int8)
        
        # Quantize input
        input_int8 = ((input - input.min()) / self.input_scale).round().clamp(0, 255).to(torch.int8)
        input_int8 = input_int8 - self.input_zero_point
        
        # INT8 matrix multiplication
        # Need to use INT32 accumulator to prevent overflow
        output_int32 = torch.matmul(
            input_int8.to(torch.int32),
            self.weight_int8.t().to(torch.int32)
        )
        
        # Dequantize output
        output = output_int32.float() * self.input_scale * self.weight_scale.t()
        
        # Add bias
        if self.bias is not None:
            output += self.bias
            
        return output
```

### Optimized INT8 with CUTLASS

```python
class CutlassINT8Linear(nn.Module):
    """INT8 linear using CUTLASS kernels."""
    
    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        try:
            import cutlass
            self.cutlass_available = True
        except:
            self.cutlass_available = False
            
        self.in_features = in_features
        self.out_features = out_features
        
        # Quantized weights in CUTLASS format
        self.register_buffer('weight_int8', None)
        self.register_buffer('weight_scale', None)
        
    def prepare_weights(self, weight: torch.Tensor):
        """Prepare weights for CUTLASS INT8 GEMM."""
        if not self.cutlass_available:
            raise RuntimeError("CUTLASS not available")
            
        # Reorder weights for tensor core layout
        weight_reordered = self._reorder_for_tensor_cores(weight)
        
        # Quantize with symmetric quantization for CUTLASS
        max_val = weight_reordered.abs().max(dim=1, keepdim=True)[0]
        scale = max_val / 127.0
        
        weight_int8 = (weight_reordered / scale).round().clamp(-128, 127).to(torch.int8)
        
        self.weight_int8 = weight_int8
        self.weight_scale = scale
    
    def _reorder_for_tensor_cores(self, weight: torch.Tensor):
        """Reorder weight layout for tensor cores."""
        # Tensor cores expect specific memory layout
        # This is a simplified version
        return weight.contiguous()
    
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        if self.cutlass_available:
            # Use CUTLASS kernel
            return self._cutlass_forward(input)
        else:
            # Fallback to PyTorch
            return self._pytorch_forward(input)
    
    def _cutlass_forward(self, input: torch.Tensor):
        """Forward using CUTLASS INT8 GEMM."""
        # Quantize input
        input_scale = input.abs().max() / 127.0
        input_int8 = (input / input_scale).round().clamp(-128, 127).to(torch.int8)
        
        # CUTLASS INT8 GEMM (pseudo-code)
        # output = cutlass.int8_gemm(input_int8, self.weight_int8)
        
        # For demonstration, use PyTorch
        output = torch.matmul(input_int8.float(), self.weight_int8.t().float())
        
        # Dequantize
        output = output * input_scale * self.weight_scale.t()
        
        return output
```

## Weight-Only Quantization

### W8A16 (Weights INT8, Activations FP16)

```python
class W8A16Linear(nn.Module):
    """Weight-only INT8 quantization."""
    
    def __init__(self, weight: torch.Tensor, bias: Optional[torch.Tensor] = None):
        super().__init__()
        self.out_features, self.in_features = weight.shape
        
        # Quantize weights at initialization
        self.register_buffer('weight_int8', None)
        self.register_buffer('weight_scale', None)
        self.quantize_weight(weight)
        
        if bias is not None:
            self.register_buffer('bias', bias)
        else:
            self.register_buffer('bias', None)
    
    def quantize_weight(self, weight: torch.Tensor):
        """Quantize weight matrix to INT8."""
        # Per-channel asymmetric quantization
        w_min = weight.min(dim=1, keepdim=True)[0]
        w_max = weight.max(dim=1, keepdim=True)[0]
        
        # Scale to INT8 range
        scale = (w_max - w_min) / 255.0
        zero_point = -w_min / scale
        
        # Quantize
        weight_int8 = ((weight - w_min) / scale).round().to(torch.uint8)
        
        self.weight_int8 = weight_int8
        self.weight_scale = scale
        self.weight_zero_point = zero_point
    
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        # Dequantize weights on-the-fly
        weight_fp16 = (self.weight_int8.float() - self.weight_zero_point) * self.weight_scale
        weight_fp16 = weight_fp16.half()
        
        # FP16 computation
        output = F.linear(input.half(), weight_fp16, self.bias)
        
        return output
```

## 4-bit Quantization

### Basic 4-bit Implementation

```python
class INT4Linear(nn.Module):
    """4-bit quantized linear layer."""
    
    def __init__(self, in_features: int, out_features: int, groupsize: int = 128):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.groupsize = groupsize
        
        # 4-bit weights packed into INT8
        n_groups = (in_features + groupsize - 1) // groupsize
        self.register_buffer('weight_int4_packed', 
                           torch.zeros((out_features, in_features // 2), dtype=torch.uint8))
        self.register_buffer('scales', torch.zeros((out_features, n_groups), dtype=torch.float16))
        self.register_buffer('zeros', torch.zeros((out_features, n_groups), dtype=torch.float16))
    
    def pack_int4(self, weight: torch.Tensor) -> torch.Tensor:
        """Pack two INT4 values into one INT8."""
        # Ensure even number of columns
        if weight.shape[1] % 2 != 0:
            weight = F.pad(weight, (0, 1))
        
        # Pack pairs of 4-bit values
        weight_int4 = weight.to(torch.uint8)
        packed = (weight_int4[:, 0::2] & 0x0F) | ((weight_int4[:, 1::2] & 0x0F) << 4)
        
        return packed
    
    def unpack_int4(self, packed: torch.Tensor) -> torch.Tensor:
        """Unpack INT8 to two INT4 values."""
        # Create output tensor
        unpacked = torch.zeros(
            (packed.shape[0], packed.shape[1] * 2), 
            dtype=torch.uint8, 
            device=packed.device
        )
        
        # Unpack
        unpacked[:, 0::2] = packed & 0x0F
        unpacked[:, 1::2] = (packed >> 4) & 0x0F
        
        return unpacked
    
    def quantize_weight(self, weight: torch.Tensor):
        """Quantize weight to 4-bit with group-wise quantization."""
        # Reshape for group-wise quantization
        weight_reshaped = weight.reshape(self.out_features, -1, self.groupsize)
        
        # Compute scale and zero per group
        w_min = weight_reshaped.min(dim=2, keepdim=True)[0]
        w_max = weight_reshaped.max(dim=2, keepdim=True)[0]
        
        scales = (w_max - w_min) / 15.0  # 4-bit -> 15 levels
        zeros = -w_min / scales
        
        # Quantize
        weight_int4 = ((weight_reshaped - w_min) / scales).round().clamp(0, 15)
        
        # Reshape and pack
        weight_int4 = weight_int4.reshape(self.out_features, -1)
        self.weight_int4_packed = self.pack_int4(weight_int4)
        self.scales = scales.squeeze(2).half()
        self.zeros = zeros.squeeze(2).half()
    
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        # Unpack weights
        weight_int4 = self.unpack_int4(self.weight_int4_packed)
        
        # Dequantize weights
        weight_fp16 = torch.zeros(
            (self.out_features, self.in_features), 
            dtype=torch.float16, 
            device=input.device
        )
        
        for g in range(self.scales.shape[1]):
            start_idx = g * self.groupsize
            end_idx = min((g + 1) * self.groupsize, self.in_features)
            
            weight_fp16[:, start_idx:end_idx] = (
                (weight_int4[:, start_idx:end_idx].float() - self.zeros[:, g:g+1]) * 
                self.scales[:, g:g+1]
            ).half()
        
        # Compute output
        output = F.linear(input, weight_fp16)
        
        return output
```

## GPTQ: Gradient-based Post-training Quantization

### GPTQ Implementation

```python
class GPTQQuantizer:
    """GPTQ: Accurate Post-Training Quantization for Generative Pre-trained Transformers"""
    
    def __init__(self, bits: int = 4, groupsize: int = 128, actorder: bool = True):
        self.bits = bits
        self.groupsize = groupsize
        self.actorder = actorder
        self.maxq = 2 ** bits - 1
        
    def quantize_block(self, weight: torch.Tensor, 
                      hessian: torch.Tensor,
                      blocksize: int = 128) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Quantize a weight matrix using GPTQ algorithm."""
        W = weight.clone()
        H = hessian.clone()
        
        rows, columns = W.shape
        device = W.device
        
        # Initialize quantized weight
        Q = torch.zeros_like(W, dtype=torch.int32)
        
        # Scales and zeros for dequantization
        scales = torch.zeros((rows, (columns + self.groupsize - 1) // self.groupsize), device=device)
        zeros = torch.zeros_like(scales)
        
        # Process in blocks
        for i in range(0, columns, blocksize):
            i_end = min(i + blocksize, columns)
            block_columns = i_end - i
            
            # Extract block
            W_block = W[:, i:i_end]
            H_block = H[i:i_end, i:i_end]
            
            # Quantize block
            Q_block = torch.zeros_like(W_block, dtype=torch.int32)
            Err = torch.zeros_like(W_block)
            
            # GPTQ algorithm
            H_inv = torch.inverse(H_block)
            
            for j in range(block_columns):
                # Quantize column
                w = W_block[:, j]
                
                # Group-wise quantization
                g_idx = (i + j) // self.groupsize
                if g_idx < scales.shape[1]:
                    # Find optimal scale and zero
                    w_min = w.min()
                    w_max = w.max()
                    
                    scale = (w_max - w_min) / self.maxq
                    zero = -w_min / scale
                    
                    scales[:, g_idx] = scale
                    zeros[:, g_idx] = zero
                    
                    # Quantize
                    q = ((w - w_min) / scale).round().clamp(0, self.maxq)
                    Q_block[:, j] = q.to(torch.int32)
                    
                    # Update error
                    w_hat = (q - zero) * scale
                    err = (w - w_hat) / H_inv[j, j]
                    
                    # Update remaining columns
                    if j < block_columns - 1:
                        W_block[:, j+1:] -= err.unsqueeze(1) @ H_inv[j, j+1:].unsqueeze(0)
            
            Q[:, i:i_end] = Q_block
        
        return Q, scales, zeros
    
    def find_optimal_permutation(self, weight: torch.Tensor, 
                                hessian: torch.Tensor) -> torch.Tensor:
        """Find column permutation to minimize quantization error."""
        if not self.actorder:
            return torch.arange(weight.shape[1])
        
        # Order by diagonal of Hessian (activation magnitude)
        H_diag = torch.diag(hessian)
        perm = torch.argsort(H_diag, descending=True)
        
        return perm
```

### GPTQ Layer Wrapper

```python
class GPTQLinear(nn.Module):
    """Linear layer with GPTQ quantization."""
    
    def __init__(self, in_features: int, out_features: int, bits: int = 4):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.bits = bits
        
        # Quantized weights
        self.register_buffer('qweight', None)
        self.register_buffer('scales', None)
        self.register_buffer('zeros', None)
        self.register_buffer('g_idx', None)  # Group indices
        
    def pack_weight(self, qweight: torch.Tensor) -> torch.Tensor:
        """Pack quantized weights for efficient storage."""
        if self.bits == 4:
            # Pack two 4-bit values into one byte
            packed = torch.zeros(
                (qweight.shape[0], qweight.shape[1] // 2), 
                dtype=torch.uint8,
                device=qweight.device
            )
            
            packed = (qweight[:, 0::2] & 0x0F) | ((qweight[:, 1::2] & 0x0F) << 4)
            return packed
        else:
            return qweight
    
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        # Unpack and dequantize weights
        if self.bits == 4:
            # Unpack 4-bit weights
            weight = self.unpack_4bit_weight()
        else:
            weight = self.qweight.float()
        
        # Dequantize
        weight_fp16 = torch.zeros(
            (self.out_features, self.in_features),
            dtype=torch.float16,
            device=input.device
        )
        
        # Apply scales and zeros group-wise
        for g in range(self.scales.shape[1]):
            group_start = self.g_idx[g]
            group_end = self.g_idx[g + 1] if g < len(self.g_idx) - 1 else self.in_features
            
            weight_fp16[:, group_start:group_end] = (
                (weight[:, group_start:group_end] - self.zeros[:, g:g+1]) * 
                self.scales[:, g:g+1]
            ).half()
        
        # Matrix multiplication
        output = F.linear(input, weight_fp16)
        
        return output
```

## AWQ: Activation-aware Weight Quantization

### AWQ Implementation

```python
class AWQQuantizer:
    """AWQ: Activation-aware Weight Quantization"""
    
    def __init__(self, bits: int = 4, group_size: int = 128):
        self.bits = bits
        self.group_size = group_size
        
    def search_scale_subset(self, weight: torch.Tensor, 
                           input_feat: torch.Tensor,
                           n_samples: int = 128) -> torch.Tensor:
        """Search for optimal scales using activation statistics."""
        # Sample input features
        if input_feat.shape[0] > n_samples:
            indices = torch.randperm(input_feat.shape[0])[:n_samples]
            input_feat = input_feat[indices]
        
        # Compute activation scales
        scales = self._compute_activation_scales(input_feat)
        
        # Grid search for optimal scale factors
        best_scales = torch.ones_like(scales)
        best_loss = float('inf')
        
        # Search space
        scale_candidates = torch.tensor([0.5, 0.75, 1.0, 1.25, 1.5])
        
        for alpha in scale_candidates:
            # Scale weights by activation magnitude
            scaled_weight = weight * (scales.pow(alpha))
            
            # Quantize
            q_weight = self.quantize_weight(scaled_weight)
            
            # Compute reconstruction error
            output_orig = F.linear(input_feat, weight)
            output_quant = F.linear(input_feat, q_weight / scales.pow(alpha))
            
            loss = (output_orig - output_quant).pow(2).mean().item()
            
            if loss < best_loss:
                best_loss = loss
                best_scales = scales.pow(alpha)
        
        return best_scales
    
    def _compute_activation_scales(self, input_feat: torch.Tensor) -> torch.Tensor:
        """Compute per-channel scales based on activation magnitude."""
        # Average absolute activation per channel
        scales = input_feat.abs().mean(dim=0)
        
        # Normalize
        scales = scales / scales.mean()
        scales = scales.clamp(min=0.01)  # Avoid numerical issues
        
        return scales
    
    def quantize_weight(self, weight: torch.Tensor) -> torch.Tensor:
        """Quantize weight with group-wise quantization."""
        # Similar to GPTQ but simpler
        n_groups = (weight.shape[1] + self.group_size - 1) // self.group_size
        
        scales = torch.zeros((weight.shape[0], n_groups))
        zeros = torch.zeros((weight.shape[0], n_groups))
        qweight = torch.zeros_like(weight, dtype=torch.int32)
        
        for g in range(n_groups):
            start = g * self.group_size
            end = min((g + 1) * self.group_size, weight.shape[1])
            
            w = weight[:, start:end]
            w_min = w.min(dim=1, keepdim=True)[0]
            w_max = w.max(dim=1, keepdim=True)[0]
            
            scale = (w_max - w_min) / (2**self.bits - 1)
            zero = -w_min / scale
            
            scales[:, g] = scale.squeeze()
            zeros[:, g] = zero.squeeze()
            
            # Quantize
            q = ((w - w_min) / scale).round().clamp(0, 2**self.bits - 1)
            qweight[:, start:end] = q.to(torch.int32)
        
        return qweight
```

## Mixed Precision Quantization

### Layer-wise Mixed Precision

```python
class MixedPrecisionModel(nn.Module):
    """Model with different quantization for different layers."""
    
    def __init__(self, base_model: nn.Module, precision_config: Dict[str, int]):
        super().__init__()
        self.precision_config = precision_config
        
        # Quantize layers based on config
        for name, module in base_model.named_modules():
            if isinstance(module, nn.Linear):
                bits = self._get_precision_for_layer(name)
                
                if bits == 8:
                    quantized = INT8Linear(module.in_features, module.out_features)
                    quantized.quantize_weights(module.weight)
                elif bits == 4:
                    quantized = INT4Linear(module.in_features, module.out_features)
                    quantized.quantize_weight(module.weight)
                else:  # Keep FP16
                    quantized = module
                
                # Replace module
                parent_name = '.'.join(name.split('.')[:-1])
                child_name = name.split('.')[-1]
                parent = base_model
                for part in parent_name.split('.'):
                    if part:
                        parent = getattr(parent, part)
                setattr(parent, child_name, quantized)
    
    def _get_precision_for_layer(self, layer_name: str) -> int:
        """Get precision for a specific layer."""
        # Example: Use 4-bit for large layers, 8-bit for small ones
        if 'mlp' in layer_name and 'down_proj' in layer_name:
            return 4  # Large MLP layers can tolerate lower precision
        elif 'attention' in layer_name and 'o_proj' in layer_name:
            return 8  # Attention output needs higher precision
        else:
            return 16  # Keep others at FP16
```

### Sensitivity-based Quantization

```python
class SensitivityAnalyzer:
    """Analyze layer sensitivity to quantization."""
    
    def __init__(self, model: nn.Module, calibration_data: torch.Tensor):
        self.model = model
        self.calibration_data = calibration_data
        
    def analyze_sensitivity(self) -> Dict[str, float]:
        """Compute sensitivity scores for each layer."""
        # Get baseline output
        with torch.no_grad():
            baseline_output = self.model(self.calibration_data)
        
        sensitivities = {}
        
        for name, module in self.model.named_modules():
            if isinstance(module, nn.Linear):
                # Temporarily quantize layer
                original_weight = module.weight.clone()
                
                # Try different bit widths
                for bits in [4, 8]:
                    # Quantize
                    if bits == 8:
                        q_weight, scale, zp = self.quantize_int8(module.weight)
                        module.weight.data = self.dequantize_int8(q_weight, scale, zp)
                    else:
                        q_weight, scale, zp = self.quantize_int4(module.weight)
                        module.weight.data = self.dequantize_int4(q_weight, scale, zp)
                    
                    # Measure impact
                    with torch.no_grad():
                        quant_output = self.model(self.calibration_data)
                    
                    # Compute error
                    error = (baseline_output - quant_output).pow(2).mean().item()
                    sensitivities[f"{name}_{bits}bit"] = error
                
                # Restore original weight
                module.weight.data = original_weight
        
        return sensitivities
```

## Quantization-Aware Training (QAT)

### QAT Implementation

```python
class QATLinear(nn.Module):
    """Linear layer with quantization-aware training."""
    
    def __init__(self, in_features: int, out_features: int, bits: int = 8):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.bits = bits
        
        # Full precision weights for training
        self.weight = nn.Parameter(torch.randn(out_features, in_features))
        self.bias = nn.Parameter(torch.zeros(out_features))
        
        # Quantization parameters (learnable)
        self.weight_scale = nn.Parameter(torch.ones(out_features, 1))
        self.weight_zero_point = nn.Parameter(torch.zeros(out_features, 1))
        
        # Activation quantization (tracked with EMA)
        self.register_buffer('input_scale', torch.tensor(1.0))
        self.register_buffer('input_zero_point', torch.tensor(0.0))
        self.momentum = 0.1
        
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        if self.training:
            # Fake quantization during training
            # Quantize weights
            weight_q = self.fake_quantize(
                self.weight, 
                self.weight_scale, 
                self.weight_zero_point,
                self.bits
            )
            
            # Track input statistics
            self.update_input_stats(input)
            
            # Quantize input
            input_q = self.fake_quantize(
                input,
                self.input_scale,
                self.input_zero_point,
                self.bits
            )
            
            # Compute with quantized values
            output = F.linear(input_q, weight_q, self.bias)
        else:
            # Real quantization during inference
            weight_q = self.quantize(
                self.weight,
                self.weight_scale,
                self.weight_zero_point,
                self.bits
            )
            
            input_q = self.quantize(
                input,
                self.input_scale,
                self.input_zero_point,
                self.bits
            )
            
            output = F.linear(input_q, weight_q, self.bias)
        
        return output
    
    def fake_quantize(self, x: torch.Tensor, scale: torch.Tensor, 
                     zero_point: torch.Tensor, bits: int) -> torch.Tensor:
        """Simulate quantization during training."""
        # Quantize
        x_q = torch.round(x / scale + zero_point)
        x_q = torch.clamp(x_q, 0, 2**bits - 1)
        
        # Dequantize
        x_dq = (x_q - zero_point) * scale
        
        # Straight-through estimator
        return x + (x_dq - x).detach()
    
    def update_input_stats(self, input: torch.Tensor):
        """Update input quantization parameters with EMA."""
        # Compute min/max
        x_min = input.detach().min()
        x_max = input.detach().max()
        
        # Compute scale and zero point
        scale = (x_max - x_min) / (2**self.bits - 1)
        zero_point = -x_min / scale
        
        # Update with momentum
        self.input_scale.mul_(1 - self.momentum).add_(scale * self.momentum)
        self.input_zero_point.mul_(1 - self.momentum).add_(zero_point * self.momentum)
```

## Production Deployment

### Quantized Model Server

```python
class QuantizedModelServer:
    """Serve quantized models efficiently."""
    
    def __init__(self, model_path: str, quantization_config: Dict):
        self.config = quantization_config
        self.model = self.load_quantized_model(model_path)
        self.optimize_for_inference()
        
    def load_quantized_model(self, path: str) -> nn.Module:
        """Load model with appropriate quantization."""
        # Load base model
        state_dict = torch.load(path, map_location='cpu')
        
        # Determine quantization type
        quant_type = self.config.get('type', 'int8')
        
        if quant_type == 'gptq':
            model = self.load_gptq_model(state_dict)
        elif quant_type == 'awq':
            model = self.load_awq_model(state_dict)
        elif quant_type == 'int8':
            model = self.load_int8_model(state_dict)
        else:
            raise ValueError(f"Unknown quantization type: {quant_type}")
        
        return model.cuda()
    
    def optimize_for_inference(self):
        """Apply inference optimizations."""
        # Fuse operations where possible
        self.model = torch.jit.script(self.model)
        
        # Enable cuDNN autotuner
        torch.backends.cudnn.benchmark = True
        
        # Use TensorRT if available
        if self.config.get('use_tensorrt', False):
            self.model = self.convert_to_tensorrt(self.model)
    
    @torch.no_grad()
    def generate(self, input_ids: torch.Tensor, max_length: int = 100):
        """Generate text with quantized model."""
        generated = input_ids
        
        for _ in range(max_length):
            # Use static shapes for better performance
            if generated.shape[1] > 2048:
                # Truncate context if too long
                generated = generated[:, -2048:]
            
            outputs = self.model(generated)
            next_token = outputs.logits[:, -1, :].argmax(dim=-1, keepdim=True)
            generated = torch.cat([generated, next_token], dim=1)
            
            if next_token.item() == self.eos_token_id:
                break
        
        return generated
```

### Benchmarking Quantized Models

```python
class QuantizationBenchmark:
    """Benchmark different quantization methods."""
    
    def __init__(self, model: nn.Module, test_data: torch.Tensor):
        self.model = model
        self.test_data = test_data
        
    def benchmark_all_methods(self) -> Dict[str, Dict]:
        methods = {
            'fp16': lambda m: m.half(),
            'int8': lambda m: self.quantize_int8(m),
            'int4': lambda m: self.quantize_int4(m),
            'gptq': lambda m: self.quantize_gptq(m),
            'awq': lambda m: self.quantize_awq(m)
        }
        
        results = {}
        
        for name, quantize_fn in methods.items():
            print(f"Benchmarking {name}...")
            
            # Quantize model
            q_model = quantize_fn(self.model.clone())
            
            # Measure metrics
            results[name] = {
                'model_size': self.get_model_size(q_model),
                'latency': self.measure_latency(q_model),
                'throughput': self.measure_throughput(q_model),
                'accuracy': self.measure_accuracy(q_model),
                'memory_usage': self.measure_memory_usage(q_model)
            }
        
        return results
    
    def get_model_size(self, model: nn.Module) -> float:
        """Get model size in MB."""
        param_size = 0
        buffer_size = 0
        
        for param in model.parameters():
            param_size += param.nelement() * param.element_size()
        
        for buffer in model.buffers():
            buffer_size += buffer.nelement() * buffer.element_size()
        
        return (param_size + buffer_size) / 1024 / 1024
    
    def measure_latency(self, model: nn.Module, num_runs: int = 100) -> float:
        """Measure average inference latency."""
        model.eval()
        
        # Warmup
        for _ in range(10):
            _ = model(self.test_data)
        
        # Measure
        torch.cuda.synchronize()
        start = time.time()
        
        for _ in range(num_runs):
            _ = model(self.test_data)
        
        torch.cuda.synchronize()
        elapsed = time.time() - start
        
        return elapsed / num_runs * 1000  # ms
```

## Best Practices

### 1. Choosing Quantization Method

```python
def select_quantization_method(model_size: float, 
                             target_speedup: float,
                             accuracy_threshold: float) -> str:
    """Select appropriate quantization method based on requirements."""
    
    # Small models (<1B params): INT8 usually sufficient
    if model_size < 1e9:
        return "int8"
    
    # Medium models (1B-10B): W8A16 or GPTQ
    elif model_size < 10e9:
        if target_speedup > 2.0:
            return "gptq"  # 4-bit for higher speedup
        else:
            return "w8a16"  # 8-bit weight-only
    
    # Large models (>10B): AWQ or GPTQ
    else:
        if accuracy_threshold > 0.99:
            return "awq"  # Better accuracy
        else:
            return "gptq"  # Better compression
```

### 2. Calibration Data Selection

```python
def prepare_calibration_data(dataset, num_samples: int = 128) -> torch.Tensor:
    """Prepare representative calibration data."""
    # Sample diverse examples
    samples = []
    
    # Stratified sampling if possible
    if hasattr(dataset, 'categories'):
        samples_per_category = num_samples // len(dataset.categories)
        for category in dataset.categories:
            category_data = dataset.filter(lambda x: x['category'] == category)
            samples.extend(random.sample(category_data, samples_per_category))
    else:
        # Random sampling
        samples = random.sample(dataset, num_samples)
    
    # Process and concatenate
    calibration_data = torch.stack([
        process_sample(sample) for sample in samples
    ])
    
    return calibration_data
```

### 3. Mixed Precision Strategy

```python
def create_mixed_precision_config(model: nn.Module, 
                                 calibration_data: torch.Tensor) -> Dict[str, int]:
    """Create layer-wise precision configuration."""
    analyzer = SensitivityAnalyzer(model, calibration_data)
    sensitivities = analyzer.analyze_sensitivity()
    
    config = {}
    
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            # Check sensitivity at different bit widths
            sens_4bit = sensitivities.get(f"{name}_4bit", float('inf'))
            sens_8bit = sensitivities.get(f"{name}_8bit", float('inf'))
            
            # Choose precision based on sensitivity
            if sens_4bit < 0.01:  # Very low sensitivity
                config[name] = 4
            elif sens_8bit < 0.01:  # Low sensitivity
                config[name] = 8
            else:  # High sensitivity
                config[name] = 16
    
    return config
```

## Conclusion

Quantization is essential for efficient LLM deployment:

1. **INT8**: 2x compression with minimal accuracy loss
2. **INT4**: 4x compression, suitable for large models
3. **GPTQ/AWQ**: State-of-the-art 4-bit quantization
4. **Mixed Precision**: Optimal accuracy/efficiency trade-off
5. **QAT**: Best accuracy but requires training

Key insights:
- Weight-only quantization (W8A16/W4A16) often provides best latency/accuracy trade-off
- Group-wise quantization essential for 4-bit
- Activation-aware methods (AWQ) improve accuracy
- Hardware support (Tensor Cores) crucial for speedup
- Careful calibration data selection important

Modern quantization enables running 70B+ parameter models on consumer GPUs while maintaining usable accuracy.