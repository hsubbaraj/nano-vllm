from dataclasses import dataclass
from enum import Enum
from typing import List, Optional
import uuid

class SequenceStatus(Enum):
    WAITING = "waiting"      # Queued for processing
    RUNNING = "running"      # Currently being processed  
    FINISHED = "finished"    # Generation complete

@dataclass
class SamplingParams:
    temperature: float = 0.0
    max_tokens: int = 256
    stop_token_ids: List[int] = None

    def __post_init__(self):
        if self.stop_token_ids is None:
            self.stop_token_ids = []

class Sequence:

    def __init__(
        self,
        prompt_tokens: List[int],
        sampling_params: SamplingParams,
    ):
        self.seq_id = str(uuid.uuid4())