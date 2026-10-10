"""Answer-prefilled generation with balanced-brace stopping and raw token credit."""
from __future__ import annotations

from types import SimpleNamespace
from uuid import uuid4

from agent_r1.evaluation.answers import final_answer_record

FINAL_ANSWER_PREFIX = r"\boxed{"
SUBMISSION_CHUNK_TOKENS = 64


def prefilled_box_end(completion: str) -> int | None:
    """End of the first completed outer box; escaped braces are math symbols."""
    depth, backslashes = 1, 0
    for index, char in enumerate(completion):
        if char == "\\":
            backslashes += 1
            continue
        escaped = backslashes % 2 == 1
        backslashes = 0
        if not escaped:
            depth += (char == "{") - (char == "}")
        if depth == 0:
            return index + 1
    return None


def prefilled_final_record(completion: str) -> dict | None:
    end = prefilled_box_end(completion)
    if end is None:
        return None
    content = completion[:end - 1].strip()
    if final_answer_record(FINAL_ANSWER_PREFIX + completion[:end]) is None:
        return None
    return {"answer": content, "method": "prefilled_boxed", "reasoning": ""}


def completed_prefix_tokens(tokenizer, token_ids: list[int]) -> int | None:
    """Retain whole sampled tokens through the box, never re-encode a response."""
    if not token_ids or prefilled_box_end(tokenizer.decode(token_ids, skip_special_tokens=True)) is None:
        return None
    low, high = 1, len(token_ids)
    while low < high:
        middle = (low + high) // 2
        if prefilled_box_end(tokenizer.decode(token_ids[:middle], skip_special_tokens=True)) is None:
            low = middle + 1
        else:
            high = middle
    return low


async def generate_prefilled_submission(server, tokenizer, *, prompt_ids: list[int],
                                        sampling_params: dict, token_limit: int):
    """Several transport requests remain one policy turn and one token budget.

    No logits mask, forced answer, or reference is used. Each continuation is
    conditioned on the exact sampled token IDs. At most one chunk is generated
    past a closing box; discarded suffix tokens remain in the budget audit.
    """
    ids, log_probs, routes = [], [], []
    consumed, requests = 0, 0
    have_log_probs = True
    request_id = uuid4().hex
    while consumed < token_limit:
        cap = min(SUBMISSION_CHUNK_TOKENS, token_limit - consumed)
        output = await server.generate(
            request_id=request_id, prompt_ids=prompt_ids + ids,
            sampling_params={**sampling_params, "max_tokens": cap},
        )
        requests += 1
        chunk = list(output.token_ids[:cap])
        consumed += len(chunk)
        if not chunk:
            break
        ids.extend(chunk)
        if output.log_probs is None:
            have_log_probs = False
        else:
            log_probs.extend(output.log_probs[:len(chunk)])
        routing = getattr(output, "routed_experts", None)
        if routing is not None:
            routes.append(routing if not routes else routing[-len(chunk):])
        kept = completed_prefix_tokens(tokenizer, ids)
        if kept is not None:
            ids, log_probs = ids[:kept], log_probs[:kept]
            break
        if len(chunk) < cap or chunk[-1] == getattr(tokenizer, "eos_token_id", None):
            break
    routing = None
    if routes:
        if len(routes) != requests:
            raise RuntimeError("Submission chunks have incomplete routing replay records")
        if hasattr(routes[0], "device"):
            import torch
            routing = torch.cat(routes, dim=0)
        else:
            import numpy as np
            routing = np.concatenate(routes, axis=0)
        routing = routing[:len(prompt_ids) + len(ids)]
    return SimpleNamespace(token_ids=ids, log_probs=log_probs if have_log_probs else None,
                           routed_experts=routing, generated_tokens=consumed,
                           generation_requests=requests, discarded_tokens=consumed - len(ids))
