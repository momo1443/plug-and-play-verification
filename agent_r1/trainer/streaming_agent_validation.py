"""Crash-resilient formal A0 validation on the shared AgentFlow path."""

from __future__ import annotations

import os
import uuid
from collections import defaultdict
from typing import Any

import numpy as np

from agent_r1.trainer.compact_validation_record import build_validation_record
from agent_r1.trainer.ppo.ray_trainer import (
    RayAgentTrainer,
    make_json_safe,
    process_validation_metrics,
)
from agent_r1.trainer.streaming_jsonl import StreamingJsonlWriter
from verl import DataProto


def _env_flag(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _step_field(non_tensor_batch: dict[str, Any], name: str, index: int, default: Any) -> Any:
    values = non_tensor_batch.get(name)
    if values is None:
        return default
    return values[index]


def _streaming_metric_keys() -> set[str] | None:
    raw = os.getenv("HOTPOTQA_STREAMING_METRIC_KEYS", "").strip()
    if not raw:
        return None
    keys = {value.strip() for value in raw.split(",") if value.strip()}
    if not keys:
        raise ValueError("HOTPOTQA_STREAMING_METRIC_KEYS must contain at least one key")
    return keys


class StreamingValidationRayAgentTrainer(RayAgentTrainer):
    """Persist one compact record per AgentFlow trajectory during validation."""

    def _validate(self):
        if not _env_flag("HOTPOTQA_STREAMING_RESULTS", True):
            return super()._validate()

        validation_dir = self.config.trainer.get("validation_data_dir", None)
        if not validation_dir:
            return super()._validate()

        stream_path = os.path.join(validation_dir, f"{self.global_steps}.jsonl")
        resume = _env_flag("HOTPOTQA_STREAMING_RESUME", False)
        fsync = _env_flag("HOTPOTQA_STREAMING_FSYNC", True)
        val_n = int(self.config.actor_rollout_ref.rollout.val_kwargs.n)
        expected_samples = len(self.val_dataloader.dataset) * val_n

        data_source_lst: list[Any] = []
        sample_inputs: list[str] = []
        sample_outputs: list[str] = []
        sample_scores: list[float] = []
        sample_uids: list[str] = []
        reward_extra_infos_dict: dict[str, list[Any]] = defaultdict(list)
        aggregate_metric_keys = _streaming_metric_keys()

        print(
            f"Streaming AgentFlow validation to {stream_path} "
            f"(resume={resume}, fsync={fsync}, expected={expected_samples})"
        )
        with StreamingJsonlWriter(stream_path, resume=resume, fsync=fsync) as writer:
            completed_samples = writer.contiguous_sample_count
            if completed_samples > expected_samples:
                raise RuntimeError(
                    f"Cannot resume {stream_path}: existing rows={completed_samples} exceed "
                    f"expected rows={expected_samples}"
                )
            if completed_samples % val_n != 0:
                raise RuntimeError(
                    f"Cannot resume {stream_path}: existing rollout count {completed_samples} "
                    f"is not divisible by val n={val_n}"
                )
            rows_to_skip = completed_samples // val_n
            if rows_to_skip:
                print(
                    f"Resuming after {completed_samples} completed rollouts; "
                    f"skipping {rows_to_skip} validation rows"
                )
            for test_data in self.val_dataloader:
                test_batch = DataProto.from_single_dict(test_data)
                source_batch_size = len(test_batch)
                if rows_to_skip:
                    if rows_to_skip < source_batch_size:
                        raise RuntimeError(
                            f"Cannot resume {stream_path}: completed prefix ends inside a validation "
                            f"batch (remaining skip={rows_to_skip}, batch={source_batch_size})"
                        )
                    rows_to_skip -= source_batch_size
                    continue
                if "uid" not in test_batch.non_tensor_batch:
                    test_batch.non_tensor_batch["uid"] = np.array(
                        [str(uuid.uuid4()) for _ in range(len(test_batch.batch))], dtype=object
                    )

                test_batch = test_batch.repeat(
                    repeat_times=self.config.actor_rollout_ref.rollout.val_kwargs.n,
                    interleave=True,
                )
                if (
                    self.config.reward_model.enable
                    and test_batch[0].non_tensor_batch["reward_model"]["style"] == "model"
                ):
                    return {}

                trajectory_count = len(test_batch)
                ground_truths = [
                    item.non_tensor_batch.get("reward_model", {}).get("ground_truth")
                    for item in test_batch
                ]
                extra_infos = [
                    item.non_tensor_batch.get("extra_info", {}) for item in test_batch
                ]
                raw_prompts = [
                    item.non_tensor_batch.get("raw_prompt") for item in test_batch
                ]
                batch_uids = list(test_batch.non_tensor_batch["uid"])

                test_gen_batch = self._get_gen_batch(test_batch)
                test_gen_batch.meta_info = {
                    "eos_token_id": self.tokenizer.eos_token_id,
                    "pad_token_id": self.tokenizer.pad_token_id,
                    "recompute_log_prob": False,
                    "do_sample": self.config.actor_rollout_ref.rollout.val_kwargs.do_sample,
                    "validate": True,
                    "global_steps": self.global_steps,
                }
                output_batch = self.async_rollout_manager.generate_sequences(test_gen_batch)
                output_batch.meta_info["validate"] = True

                reward_result = self._compute_or_extract_reward(
                    output_batch,
                    reward_fn=self.val_reward_fn,
                    return_dict=True,
                )
                reward_tensor = reward_result["reward_tensor"]
                step_scores = reward_tensor.sum(-1).detach().cpu().tolist()
                reward_extra_info = reward_result.get("reward_extra_info", {})
                step_inputs = self.tokenizer.batch_decode(
                    output_batch.batch["input_ids"], skip_special_tokens=True
                )
                step_outputs = self.tokenizer.batch_decode(
                    output_batch.batch["responses"], skip_special_tokens=True
                )
                num_steps = [int(value) for value in output_batch.meta_info["num_steps"]]
                if len(num_steps) != trajectory_count or sum(num_steps) != len(step_scores):
                    raise ValueError(
                        f"AgentFlow trajectory shape mismatch: num_steps={num_steps}, "
                        f"step_scores={len(step_scores)}, trajectories={trajectory_count}"
                    )

                start = 0
                batch_trajectory_scores: list[float] = []
                batch_trajectory_reward_info: dict[str, list[Any]] = defaultdict(list)
                for trajectory_index, step_count in enumerate(num_steps):
                    end = start + step_count
                    last = end - 1
                    trajectory_score = float(sum(step_scores[start:end]))
                    batch_trajectory_scores.append(trajectory_score)

                    for key, values in reward_extra_info.items():
                        if aggregate_metric_keys is not None and key not in aggregate_metric_keys:
                            continue
                        batch_trajectory_reward_info[key].append(
                            make_json_safe(values[last])
                        )

                    record = build_validation_record(
                        sample_index=completed_samples + len(sample_scores) + trajectory_index,
                        raw_prompt=raw_prompts[trajectory_index],
                        decoded_input=step_inputs[0 if step_count == 0 else start],
                        output_text=step_outputs[last],
                        ground_truth=ground_truths[trajectory_index],
                        score=trajectory_score,
                        extra_info=extra_infos[trajectory_index],
                        thinking_mode=_step_field(
                            output_batch.non_tensor_batch, "thinking_mode", last, "unknown"
                        ),
                        thinking_steps=_step_field(
                            output_batch.non_tensor_batch, "qwen_thinking", last, []
                        ),
                        force_first_search=_step_field(
                            output_batch.non_tensor_batch, "force_first_search", last, True
                        ),
                        final_answer_protocol=_step_field(
                            output_batch.non_tensor_batch,
                            "final_answer_protocol",
                            last,
                            None,
                        ),
                        evidence_schema_version=_step_field(
                            output_batch.non_tensor_batch,
                            "evidence_schema_version",
                            last,
                            "unknown",
                        ),
                        search_steps=_step_field(
                            output_batch.non_tensor_batch, "search_steps", last, []
                        ),
                        local_reasoning_transitions=_step_field(
                            output_batch.non_tensor_batch,
                            "local_reasoning_transitions",
                            last,
                            [],
                        ),
                        executed_queries=_step_field(
                            output_batch.non_tensor_batch,
                            "executed_search_queries",
                            last,
                            [],
                        ),
                        num_turns=step_count,
                        sample_key=_step_field(
                            output_batch.non_tensor_batch, "sample_key", last, None
                        ),
                        dataset_question_id=_step_field(
                            output_batch.non_tensor_batch,
                            "dataset_question_id",
                            last,
                            None,
                        ),
                        official_qid=_step_field(
                            output_batch.non_tensor_batch, "official_qid", last, None
                        ),
                        gold_evidence_ids=_step_field(
                            output_batch.non_tensor_batch,
                            "gold_evidence_ids",
                            last,
                            [],
                        ),
                        unresolved_gold_facts=_step_field(
                            output_batch.non_tensor_batch,
                            "unresolved_gold_facts",
                            last,
                            [],
                        ),
                        evidence_metrics=_step_field(
                            output_batch.non_tensor_batch,
                            "evidence_metrics",
                            last,
                            {},
                        ),
                    )
                    writer.append(record)
                    sample_inputs.append(step_inputs[start])
                    sample_outputs.append(step_outputs[last])
                    start = end

                sample_scores.extend(batch_trajectory_scores)
                sample_uids.extend(batch_uids)
                reward_extra_infos_dict["reward"].extend(batch_trajectory_scores)
                for key, values in batch_trajectory_reward_info.items():
                    reward_extra_infos_dict[key].extend(values)
                data_source_lst.append(
                    test_batch.non_tensor_batch.get(
                        "data_source", ["unknown"] * trajectory_count
                    )
                )
                print(
                    f"Streaming AgentFlow validation progress: "
                    f"{writer.count}/{expected_samples} -> {stream_path}"
                )

            if rows_to_skip:
                raise RuntimeError(
                    f"Cannot resume {stream_path}: {rows_to_skip} completed validation rows were "
                    "not present in the current dataloader"
                )
            if writer.count != expected_samples:
                raise RuntimeError(
                    f"Streaming validation ended at {writer.count}/{expected_samples} rows"
                )

        self._maybe_log_val_generations(
            inputs=sample_inputs, outputs=sample_outputs, scores=sample_scores
        )
        for key, values in reward_extra_infos_dict.items():
            if values and len(values) != len(sample_scores):
                raise ValueError(
                    f"Validation metric {key!r} has {len(values)} values for "
                    f"{len(sample_scores)} trajectories"
                )

        data_sources = np.concatenate(data_source_lst, axis=0)
        grouped_metrics = process_validation_metrics(
            data_sources, sample_uids, reward_extra_infos_dict
        )
        metric_dict: dict[str, Any] = {}
        for data_source, variables in grouped_metrics.items():
            core_variable = "acc" if "acc" in variables else "reward"
            for variable_name, metrics in variables.items():
                n_max = max(
                    int(name.split("@")[-1].split("/")[0]) for name in metrics
                )
                for metric_name, metric_value in metrics.items():
                    if (
                        variable_name == core_variable
                        and any(
                            metric_name.startswith(prefix)
                            for prefix in ("mean", "maj", "best")
                        )
                        and f"@{n_max}" in metric_name
                    ):
                        section = "val-core"
                    else:
                        section = "val-aux"
                    metric_dict[
                        f"{section}/{data_source}/{variable_name}/{metric_name}"
                    ] = make_json_safe(metric_value)
        return metric_dict
