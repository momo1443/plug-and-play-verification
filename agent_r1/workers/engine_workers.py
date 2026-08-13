# Copyright 2025 Agent-R1 Teams
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Thin new-engine worker wrappers that swap in Agent-R1 local losses.
"""

import os
from functools import partial
from itertools import chain

import torch
from codetiming import Timer
from tensordict import NonTensorData, TensorDict

from agent_r1.workers.utils.losses import ppo_loss
from verl.single_controller.base.decorator import Dispatch, make_nd_compute_dataproto_dispatch_fn, register
from verl.utils import tensordict_utils as tu
from verl.utils import torch_functional as verl_F
from verl.utils.config import omega_conf_to_dataclass
from verl.utils.dataset.dataset_utils import DatasetPadMode
from verl.utils.py_functional import append_to_dict
from verl.workers.config import ActorConfig
from verl.workers.engine import utils as engine_utils
from verl.workers.engine_workers import ActorRolloutRefWorker as VerlActorRolloutRefWorker
from verl.workers.engine_workers import TrainingWorker as VerlTrainingWorker

_ORIGINAL_PREPARE_MICRO_BATCHES = engine_utils.prepare_micro_batches
_ORIGINAL_INDEX_SELECT_TENSOR_DICT = tu.index_select_tensor_dict
_ORIGINAL_ENTROPY_FROM_LOGITS = verl_F.entropy_from_logits


def _entropy_from_logits_with_row_chunking(logits: torch.Tensor, chunk_rows: int) -> torch.Tensor:
    """Compute the unchanged exact entropy formula in bounded token-row chunks."""

    if chunk_rows <= 0:
        raise ValueError(f"chunk_rows must be positive, got {chunk_rows}")
    if logits.ndim == 0:
        raise ValueError("entropy logits must include a vocabulary dimension")

    flat_logits = logits.reshape(-1, logits.shape[-1])
    if flat_logits.shape[0] <= chunk_rows:
        return _ORIGINAL_ENTROPY_FROM_LOGITS(logits)

    chunks = [
        _ORIGINAL_ENTROPY_FROM_LOGITS(flat_logits[start : start + chunk_rows])
        for start in range(0, flat_logits.shape[0], chunk_rows)
    ]
    return torch.cat(chunks, dim=0).reshape(logits.shape[:-1])


def _install_entropy_row_chunking_patch() -> None:
    raw_chunk_rows = os.environ.get("AGENT_R1_ENTROPY_CHUNK_ROWS")
    if raw_chunk_rows is None:
        return
    try:
        chunk_rows = int(raw_chunk_rows)
    except ValueError as exc:
        raise ValueError(
            "AGENT_R1_ENTROPY_CHUNK_ROWS must be a positive integer, "
            f"got {raw_chunk_rows!r}"
        ) from exc
    if chunk_rows <= 0:
        raise ValueError(
            "AGENT_R1_ENTROPY_CHUNK_ROWS must be positive, "
            f"got {chunk_rows}"
        )
    verl_F.entropy_from_logits = partial(
        _entropy_from_logits_with_row_chunking, chunk_rows=chunk_rows
    )


def _select_jagged_nested(tensor: torch.Tensor, indices: list[int]) -> torch.Tensor:
    """Select ragged rows of a jagged NestedTensor without torch's buggy unbind.

    verl 0.8 stores every actor field as a jagged NestedTensor. torch's
    ``NestedTensor.unbind()`` (used by ``index_select_tensor_dict``) runs a
    ``_lengths[i] <= ragged_dim_size`` check against a *cached* ``max_seqlen``
    that goes stale after the batch is assembled via cat/narrow, so at main
    scale it raises ``RuntimeError: Expected cond to be True, but got False``.
    We instead slice the backing ``values()`` with the real ``offsets()`` and
    rebuild the jagged tensor with freshly recomputed offsets, which restores
    correct metadata and never touches the stale cache.
    """

    values = tensor.values()
    offsets = tensor.offsets()
    starts = offsets[:-1].tolist()
    ends = offsets[1:].tolist()
    # ``values`` stores the jagged sample dimension at ``ragged_idx - 1``:
    # text tensors use ``values[total_tokens]`` (ragged_idx=1), while Qwen3.5
    # mRoPE position ids use ``values[4, total_tokens]`` (ragged_idx=2).
    # Slicing dimension 0 unconditionally corrupts the latter by retaining all
    # tokens while advertising offsets for only the selected rows.
    ragged_idx = getattr(tensor, "_ragged_idx", tensor.dim() - 1)
    values_ragged_dim = ragged_idx - 1
    total_jagged_length = int(offsets[-1].item())
    if values.shape[values_ragged_dim] != total_jagged_length:
        # TensorDict/Ray serialization can reset the private ``_ragged_idx``
        # field to 1 while preserving values=[4, total_tokens] and the real
        # offsets. Recover the only values dimension compatible with offsets.
        compatible_dims = [
            dim for dim, size in enumerate(values.shape) if size == total_jagged_length
        ]
        if len(compatible_dims) != 1:
            raise ValueError(
                "Cannot infer jagged values dimension from serialized metadata: "
                f"{values.shape=}, {total_jagged_length=}, {ragged_idx=}"
            )
        values_ragged_dim = compatible_dims[0]
        ragged_idx = values_ragged_dim + 1
    parts = [values.narrow(values_ragged_dim, starts[i], ends[i] - starts[i]) for i in indices]
    if parts:
        new_values = torch.cat(parts, dim=values_ragged_dim)
    else:
        new_values = values.narrow(values_ragged_dim, 0, 0)
    lengths = torch.tensor(
        [ends[i] - starts[i] for i in indices], dtype=offsets.dtype, device=offsets.device
    )
    zero = torch.zeros(1, dtype=offsets.dtype, device=offsets.device)
    new_offsets = torch.cat([zero, torch.cumsum(lengths, dim=0)])
    selected = torch.nested.nested_tensor_from_jagged(new_values, new_offsets)
    # PyTorch constructs jagged tensors with ragged_idx=1 by default. Preserve
    # the original layout so downstream Qwen3.5 code sees [batch, 4, seq_len].
    selected._ragged_idx = ragged_idx
    return selected


def _safe_index_select_tensor_dict(batch, indices):
    """Drop-in for verl's ``index_select_tensor_dict`` that fixes jagged unbind."""

    if isinstance(indices, list):
        indices = torch.tensor(indices)
    assert indices.dim() == 1, "indices must be a 1D tensor"
    if batch is None:
        return None

    index_list = [int(i) for i in indices.tolist()]
    data_dict = {}
    for key, tensor in batch.items():
        if isinstance(tensor, torch.Tensor) and not tensor.is_nested:
            data_dict[key] = tensor[indices]
        elif isinstance(tensor, torch.Tensor) and tensor.is_nested:
            if getattr(tensor, "layout", None) == torch.jagged:
                data_dict[key] = _select_jagged_nested(tensor, index_list)
            else:
                # Non-jagged nested tensors are not produced by verl 0.8's engine;
                # defer to the upstream implementation if one ever appears.
                data_dict[key] = _ORIGINAL_INDEX_SELECT_TENSOR_DICT(
                    batch.select(key), indices
                )[key]
        else:
            data_dict[key] = tensor[indices] if getattr(tensor, "shape", None) else tensor
    return TensorDict(source=data_dict, batch_size=indices.shape[0])


def _prepare_micro_batches(
    data: TensorDict,
    dp_group=None,
    num_batches_divided_by=None,
    same_micro_num_in_dp=True,
    min_num_micro_batch=None,
    use_dynamic_bsz_balance=True,
):
    # Keep verl's dynamic path unchanged. For static micro-batching, allow a short
    # tail batch and sync the micro-batch count across DP ranks before slicing.
    use_dynamic_bsz = tu.get_non_tensor_data(data=data, key="use_dynamic_bsz", default=True)
    if use_dynamic_bsz:
        return _ORIGINAL_PREPARE_MICRO_BATCHES(
            data=data,
            dp_group=dp_group,
            num_batches_divided_by=num_batches_divided_by,
            same_micro_num_in_dp=same_micro_num_in_dp,
            min_num_micro_batch=min_num_micro_batch,
            use_dynamic_bsz_balance=use_dynamic_bsz_balance,
        )

    micro_batch_size_per_gpu = data["micro_batch_size_per_gpu"]
    assert micro_batch_size_per_gpu > 0, f"micro_batch_size_per_gpu must be positive, got {micro_batch_size_per_gpu}"
    num_micro_batches = (len(data) + micro_batch_size_per_gpu - 1) // micro_batch_size_per_gpu

    if torch.distributed.is_initialized() and same_micro_num_in_dp:
        num_micro_batches_tensor = torch.tensor([num_micro_batches], dtype=torch.long, device=data["input_ids"].device)
        torch.distributed.all_reduce(num_micro_batches_tensor, op=torch.distributed.ReduceOp.MAX, group=dp_group)
        num_micro_batches = int(num_micro_batches_tensor.cpu().item())

    if num_batches_divided_by is not None:
        num_micro_batches = ((num_micro_batches + num_batches_divided_by - 1) // num_batches_divided_by) * (
            num_batches_divided_by
        )

    if num_micro_batches > len(data):
        raise ValueError(
            f"num_micro_batches ({num_micro_batches}) must be <= local batch size ({len(data)}) "
            "when use_dynamic_bsz is disabled"
        )

    micro_batches = [
        tu.index_select_tensor_dict(
            data,
            list(
                range(
                    micro_batch_id * len(data) // num_micro_batches,
                    (micro_batch_id + 1) * len(data) // num_micro_batches,
                )
            ),
        )
        for micro_batch_id in range(num_micro_batches)
    ]
    return micro_batches, None


def _install_prepare_micro_batches_patch() -> None:
    from verl.workers.engine.fsdp import transformer_impl as fsdp_transformer_impl

    engine_utils.prepare_micro_batches = _prepare_micro_batches
    fsdp_transformer_impl.prepare_micro_batches = _prepare_micro_batches
    # Both mini-batch (train_mini_batch) and micro-batch (_prepare_micro_batches)
    # selection go through tu.index_select_tensor_dict; patch the module attribute
    # so every call site uses the jagged-safe implementation.
    tu.index_select_tensor_dict = _safe_index_select_tensor_dict


def _nested_valid_prefixes(padded: torch.Tensor, input_ids: torch.Tensor) -> torch.Tensor:
    """Restore full valid-token jagged rows from a padded fused-head output."""

    if not input_ids.is_nested:
        raise TypeError("input_ids must be a jagged NestedTensor")
    if padded.ndim < 2:
        raise ValueError(f"padded fused output must have at least two dimensions, got {padded.shape}")
    if padded.shape[0] != input_ids.shape[0]:
        raise ValueError(
            f"fused output batch size {padded.shape[0]} does not match input batch size {input_ids.shape[0]}"
        )

    seq_lengths = input_ids.offsets().diff().to(device=padded.device, dtype=torch.int64)
    if seq_lengths.numel() and int(seq_lengths.max().item()) > padded.shape[1]:
        raise ValueError(
            f"fused output sequence length {padded.shape[1]} is shorter than valid input length "
            f"{int(seq_lengths.max().item())}"
        )
    rows = [padded[index, : int(length.item())] for index, length in enumerate(seq_lengths)]
    values = torch.cat(rows, dim=0) if rows else padded.new_empty((0, *padded.shape[2:]))
    zero = torch.zeros(1, dtype=torch.int64, device=padded.device)
    offsets = torch.cat([zero, torch.cumsum(seq_lengths, dim=0)])
    return torch.nested.nested_tensor_from_jagged(values, offsets)


def _fused_token_output_as_padded(output: torch.Tensor, batch_size: int) -> torch.Tensor:
    """Normalize Qwen3.5 fused token scalars to ``[batch, padded_seq]``."""

    if output.ndim == 2 and output.shape[0] == batch_size:
        return output
    if output.ndim not in {1, 2}:
        raise ValueError(f"unexpected fused token output shape: {output.shape}")
    if output.numel() % batch_size != 0:
        raise ValueError(
            f"fused token output with {output.numel()} values is not divisible by batch size {batch_size}"
        )
    return output.reshape(batch_size, output.numel() // batch_size)


def _install_fused_no_padding_output_patch() -> None:
    """Make fused FSDP outputs compatible with Agent-R1's jagged PPO loss.

    Upstream verl 0.8 slices a dense response window when fused kernels are
    enabled with ``use_remove_padding=False``. Agent-R1's PPO loss instead
    consumes full valid-token jagged rows so it can align variable prompt and
    response lengths. Restore that representation without packing samples.
    """

    from verl.workers.engine.fsdp.transformer_impl import FSDPEngineWithLMHead

    current = FSDPEngineWithLMHead.prepare_model_outputs
    if getattr(current, "_agent_r1_fused_no_padding", False):
        return

    original = current

    def prepare_model_outputs(self, output, output_args, micro_batch, logits_processor_func):
        use_fused_kernels = tu.get_non_tensor_data(
            data=micro_batch, key="use_fused_kernels", default=False
        )
        use_remove_padding = tu.get_non_tensor_data(
            data=micro_batch, key="use_remove_padding", default=True
        )
        pad_mode = tu.get_non_tensor_data(
            data=micro_batch, key="pad_mode", default=DatasetPadMode.NO_PADDING
        )
        if not use_fused_kernels or use_remove_padding:
            return original(self, output, output_args, micro_batch, logits_processor_func)
        if pad_mode != DatasetPadMode.NO_PADDING:
            raise NotImplementedError(f"pad_mode {pad_mode} not supported")

        input_ids = micro_batch["input_ids"]
        batch_size = input_ids.shape[0]
        log_probs = _fused_token_output_as_padded(output.log_probs, batch_size)
        model_output = {"log_probs": _nested_valid_prefixes(log_probs, input_ids)}
        calculate_entropy = tu.get_non_tensor_data(
            data=micro_batch, key="calculate_entropy", default=False
        )
        if calculate_entropy:
            entropy = _fused_token_output_as_padded(output.entropy, batch_size)
            model_output["entropy"] = _nested_valid_prefixes(entropy, input_ids)
        calculate_sum_pi_squared = tu.get_non_tensor_data(
            data=micro_batch, key="calculate_sum_pi_squared", default=False
        )
        if calculate_sum_pi_squared:
            raise NotImplementedError(
                "calculate_sum_pi_squared=True is not supported with use_fused_kernels=True"
            )
        distillation_use_topk = tu.get_non_tensor_data(
            data=micro_batch, key="distillation_use_topk", default=False
        )
        if distillation_use_topk:
            raise NotImplementedError(
                "distillation top-k is not supported by Agent-R1's fused/no-remove-padding adapter"
            )
        return model_output

    prepare_model_outputs._agent_r1_fused_no_padding = True
    FSDPEngineWithLMHead.prepare_model_outputs = prepare_model_outputs


_install_entropy_row_chunking_patch()
_install_prepare_micro_batches_patch()
_install_fused_no_padding_output_patch()


class TrainingWorker(VerlTrainingWorker):
    @register(dispatch_mode=make_nd_compute_dataproto_dispatch_fn(mesh_name="train"), blocking=False)
    def train_mini_batch(self, data: TensorDict) -> TensorDict:
        if "mini_batch_id" not in data.keys():
            return super().train_mini_batch(data)

        disable_auto_offload = tu.pop(data, key="disable_auto_offload", default=False)
        mini_batch_size = tu.pop(data, key="mini_batch_size", default=None)
        num_mini_batch = tu.pop(data, key="num_mini_batch", default=None)
        epochs = tu.pop(data, key="epochs", default=1)
        seed = tu.pop(data, key="seed", default=42)
        dataloader_kwargs = tu.pop(data, key="dataloader_kwargs", default={})
        mini_batch_ids = data.pop("mini_batch_id").to(dtype=torch.long)
        mini_batch_global_sizes = data.pop("mini_batch_global_size").to(dtype=torch.long)
        mini_batch_global_token_nums = data.pop("mini_batch_global_token_num").to(dtype=torch.long)

        assert mini_batch_size is not None or num_mini_batch is not None
        assert dataloader_kwargs.keys() <= {"shuffle"}, f"Unsupported dataloader_kwargs: {dataloader_kwargs.keys()}"

        if num_mini_batch is not None:
            num_mini_batch = int(num_mini_batch)
            unique_mini_batch_ids = torch.arange(num_mini_batch, dtype=torch.long)
        else:
            unique_mini_batch_ids = torch.unique(mini_batch_ids, sorted=True).cpu()
        total_num_iterations = len(unique_mini_batch_ids) * epochs
        shuffle = dataloader_kwargs.get("shuffle", False)

        with (
            self.engine.train_mode(disable_auto_offload=disable_auto_offload),
            Timer(name="train_batch", logger=None),
        ):
            output_lst = []
            iteration_idx = 0
            for epoch in range(epochs):
                epoch_mini_batch_ids = unique_mini_batch_ids
                if shuffle:
                    generator = torch.Generator()
                    generator.manual_seed(seed + epoch)
                    permutation = torch.randperm(len(epoch_mini_batch_ids), generator=generator)
                    epoch_mini_batch_ids = epoch_mini_batch_ids[permutation]

                for mini_batch_id in epoch_mini_batch_ids:
                    indices = torch.nonzero(mini_batch_ids.cpu() == mini_batch_id, as_tuple=False).flatten()
                    mini_batch_td = tu.index_select_tensor_dict(data, indices)

                    global_token_num = mini_batch_global_token_nums[indices[0]]
                    global_token_num = global_token_num[global_token_num > 0].tolist()
                    global_batch_size = int(mini_batch_global_sizes[indices[0]].item())

                    tu.assign_non_tensor(
                        mini_batch_td,
                        global_token_num=NonTensorData(global_token_num),
                        global_batch_size=global_batch_size,
                        update_lr_scheduler=iteration_idx == total_num_iterations - 1,
                        disable_auto_offload=True,
                    )
                    output_lst.append(self.train_batch(mini_batch_td))
                    iteration_idx += 1

            if self.engine.is_mp_src_rank_with_outputs():
                actor_output = [tu.get(output, "metrics") for output in output_lst]
                metrics = {}
                for output in actor_output:
                    for key, val in output.items():
                        if isinstance(val, list):
                            output[key] = list(chain.from_iterable(val))
                    append_to_dict(metrics, output)

                output = tu.get_tensordict(tensor_dict={}, non_tensor_dict={"metrics": metrics}).cpu()
            else:
                output = None
        return output


class ActorRolloutRefWorker(VerlActorRolloutRefWorker):
    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def init_model(self):
        import verl.workers.engine_workers as upstream_engine_workers

        original_training_worker = upstream_engine_workers.TrainingWorker
        upstream_engine_workers.TrainingWorker = TrainingWorker
        try:
            super().init_model()
        finally:
            upstream_engine_workers.TrainingWorker = original_training_worker

        if "actor" in self.role:
            actor_config: ActorConfig = omega_conf_to_dataclass(self.config.actor)
            actor_config.model_config = self.config.model
            self.loss_fn = partial(ppo_loss, config=actor_config)
            self.actor.set_loss_fn(self.loss_fn)
