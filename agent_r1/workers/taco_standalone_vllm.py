"""TACO-only standalone vLLM server with an explicit worker memory ceiling."""

from __future__ import annotations

import ray
import torch

from verl.workers.rollout.llm_server import LLMServerManager
from verl.workers.rollout.vllm_rollout.utils import vLLMColocateWorkerExtension
from verl.workers.rollout.vllm_rollout.vllm_async_server import vLLMHttpServer, vLLMReplica


class TacoStandaloneVLLMWorkerExtension(vLLMColocateWorkerExtension):
    """Restore a usable PyTorch allocation fraction before vLLM loads weights."""

    def __new__(cls, **kwargs):
        local_rank = int(kwargs["local_rank"])
        torch.cuda.set_device(local_rank)
        torch.cuda.set_per_process_memory_fraction(0.95, local_rank)
        return super().__new__(cls, **kwargs)


class TacoStandaloneVLLMHttpServer(vLLMHttpServer):
    def _get_worker_extension_cls(self) -> str:
        return "agent_r1.workers.taco_standalone_vllm.TacoStandaloneVLLMWorkerExtension"

    async def launch_server(self, *args, **kwargs):
        # Direct vLLM inference works on this host without verl's runtime
        # allocator mutation. Keep the standalone server on that allocator path.
        import verl.utils.device as device_utils

        original_set_expandable_segments = device_utils.set_expandable_segments
        device_utils.set_expandable_segments = lambda enabled: None
        try:
            return await super().launch_server(*args, **kwargs)
        finally:
            device_utils.set_expandable_segments = original_set_expandable_segments


class TacoStandaloneVLLMReplica(vLLMReplica):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.server_class = ray.remote(TacoStandaloneVLLMHttpServer)


class TacoStandaloneLLMServerManager(LLMServerManager):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.rollout_replica_class = TacoStandaloneVLLMReplica
