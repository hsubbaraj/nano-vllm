import atexit
from dataclasses import fields
from time import perf_counter
from tqdm.auto import tqdm
from transformers import AutoTokenizer
import torch.multiprocessing as mp

from nanovllm.config import Config
from nanovllm.sampling_params import SamplingParams
from nanovllm.engine.sequence import Sequence
from nanovllm.engine.scheduler import Scheduler
from nanovllm.engine.model_runner import ModelRunner


class LLMEngine:

    #TODO: type def in the future
    def __init__(self, model, **kwargs):
        config_fields = {field.name for field in fields(Config)}
        config_kwargs = {k: v for k, v in kwargs.items() if k in config_fields}
        # update the config with the kwargs
        config = Config(model, **config_kwargs)
        
        # process list and event list
        self.ps = []
        self.events = []
        ctx = mp.get_context("spawn")

        # TODO: improvement - spawn worker thread even for rank 0 worker
        for i in range(1, config.tensor_parallel_size):
            event = ctx.Event()
            process = ctx.Process(target=ModelRunner, args=(config, i, event))
            process.start()
            self.events.append(event)
            self.ps.append(process)
        
        self.tokenizer = AutoTokenizer.from_pretrained(config.model, use_fast=True)
        config.eos = self.tokenizer.eos_token_id

        self.model_runner = ModelRunner(config, 0, self.events)
        self.scheduler = Scheduler(config)

        atexit.register(self.exit)

        
    def exit(self):
        self.model_runner.call("exit")
        del self.model_runner
        for p in self.ps:
            p.join()
    
    # TODO: add request id
    # 1. encode prompt if not already encoded???
    # 2. Create a sequence object
    # 3. Add sequence to the scheduler
    def add_request(self, prompt: str | List[int], sampling_params: SamplingParams):
        if isinstance(prompt, str):
            prompt = self.tokenizer.encode(prompt)
        seq = Sequence(prompt, sampling_params)
        self.scheduler.add_request(seq)

    
    def step(self):
        seqs, is_prefill = self.scheduler.schedule()
        token_ids = self.model_runner.call("run", seqs, is_prefill)
        self.scheduler.post_process(seqs, token_ids)
        outputs = [(seq.seq_id, seq.completion_token_ids) for seq in seqs if seq.is_finished]
        num_tokens = sum(len(seq) for seq in seqs) if is_prefill else -len(seqs)
        return outputs, num_tokens
    
    def is_finished(self):
        return self.scheduler.is_finished()
    
    def generate(
        self,
        prompts: List[str] | List[List[int]],
        sampling_params: SamplingParams | List[SamplingParams],
        use_tqdm: bool = False,
    ):
        # Apply the same sampling params to all prompts if not provided
        if not isinstance(prompts, list):
            sampling_params = [sampling_params] * len(prompts)
        elif len(sampling_params) != len(prompts):
            raise ValueError("Number of prompts and sampling params must match")
        
        for prompt, sp in zip(prompts, sampling_params):
            self.add_request(prompt, sp)
        
        output_map = {}
        prefill_throughput = 0
        decode_throughput = 0

        # Loop through the scheduler until all requests are finished
        while not self.is_finished():
            t = perf_counter()
            output, num_tokens = self.step()

            for seq_id, token_ids in output:
                output_map[seq_id] = token_ids
        
        # sort the output by sequence id... is this necessary?
        outputs = [output_map[seq_id] for seq_id in sorted(output_map)]
        # Decode token_ids to text
        outputs = [{"text": self.tokenizer.decode(token_ids)} for token_ids in outputs]

        return outputs