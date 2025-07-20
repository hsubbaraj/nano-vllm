from dataclasses import dataclass, field
from typing import Optional

@dataclass
class EngineConfig:
    """Central configuration for inference engine"""
    
    # Model configuration
    model_path: str
    dtype: str = "float16"
    trust_remote_code: bool = False

    # Memory management  
    block_size: int = 16           # Tokens per memory block
    max_num_blocks: int = 1024     # Total available blocks
    gpu_memory_utilization: float = 0.9  # Fraction of GPU memory to use

    # Sequence configuration
    max_num_seqs: int = 32 # max number of sequences we can run in a batch
    max_num_batched_tokens: int = 2048 # max tokens per batch

    # Parallelism configurtion
    tensor_parallel_size: int = 1 # how many GPUs to shard tensors on

    # Performance configuration
    enable_cuda_graph: bool = True
    enable_prefix_caching: bool = True

    # Derived props
    vocab_size: int = field(init=False)
    hidden_size: int = field(init=False)
    num_layers: int = field(init=False)

    def __post_init__(self):
        
        assert self.block_size > 0, "Block size must be positive int"
        assert 0 < self.gpu_memory_utilization <= 1, "GPU mem must be (0, 1]"
        assert self.tensor_parallel_size >= 1 and self.tensor_parallel_size % 2 == 0 , "Tensor parallel size must be >1 and divisible by 2"
        self._load_model_config()

    def _load_model_config(self):

        from transformers import AutoConfig

        hf_config = AutoConfig.from_pretrained(
            self.model_path,
            trust_remote_code=self.trust_remote_code
        )

        # Model config
        self.vocab_size = hf_config.vocab_size
        self.hidden_size = hf_config.hidden_size
        self.num_layers = hf_config.num_hidden_layers

       