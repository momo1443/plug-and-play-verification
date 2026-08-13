import torch
from tensordict import TensorDict

from agent_r1.workers.engine_workers import (
    _entropy_from_logits_with_row_chunking,
    _fused_token_output_as_padded,
    _nested_valid_prefixes,
    _safe_index_select_tensor_dict,
    _select_jagged_nested,
)


def _mrope_position_ids(length: int, base: int) -> torch.Tensor:
    return torch.stack([torch.arange(length) + base + channel * 10_000 for channel in range(4)])


def test_select_jagged_nested_preserves_qwen35_mrope_layout():
    samples = [
        _mrope_position_ids(5, 100),
        _mrope_position_ids(3, 200),
        _mrope_position_ids(7, 300),
    ]
    position_ids = torch.nested.as_nested_tensor(samples, layout=torch.jagged)

    # TensorDict/Ray can reset this private field while leaving values/offsets intact.
    position_ids._ragged_idx = 1

    selected = _select_jagged_nested(position_ids, [2, 0])

    assert selected.dim() == 3
    assert selected._ragged_idx == 2
    assert selected.values().shape == (4, 12)
    selected_samples = selected.unbind()
    assert torch.equal(selected_samples[0], samples[2])
    assert torch.equal(selected_samples[1], samples[0])


def test_safe_index_select_keeps_input_and_mrope_lengths_aligned():
    lengths = [5, 3, 7]
    input_samples = [torch.arange(length) + sample * 100 for sample, length in enumerate(lengths)]
    position_samples = [
        _mrope_position_ids(length, sample * 100) for sample, length in enumerate(lengths)
    ]
    batch = TensorDict(
        {
            "input_ids": torch.nested.as_nested_tensor(input_samples, layout=torch.jagged),
            "position_ids": torch.nested.as_nested_tensor(position_samples, layout=torch.jagged),
        },
        batch_size=3,
    )

    selected = _safe_index_select_tensor_dict(batch, torch.tensor([2, 0]))

    selected_input_ids = selected["input_ids"].unbind()
    selected_position_ids = selected["position_ids"].unbind()
    assert [tensor.shape[-1] for tensor in selected_input_ids] == [7, 5]
    assert [tensor.shape[-1] for tensor in selected_position_ids] == [7, 5]
    assert selected["position_ids"]._ragged_idx == 2
    assert torch.equal(selected_position_ids[0], position_samples[2])
    assert torch.equal(selected_position_ids[1], position_samples[0])


def test_entropy_row_chunking_preserves_values_and_gradients():
    torch.manual_seed(7)
    expected_logits = torch.randn(2, 7, 31, dtype=torch.float64, requires_grad=True)
    chunked_logits = expected_logits.detach().clone().requires_grad_(True)

    probabilities = torch.nn.functional.softmax(expected_logits, dim=-1)
    expected = torch.logsumexp(expected_logits, dim=-1) - torch.sum(
        probabilities * expected_logits, dim=-1
    )
    actual = _entropy_from_logits_with_row_chunking(chunked_logits, chunk_rows=3)

    torch.testing.assert_close(actual, expected, rtol=1e-12, atol=1e-12)
    expected.sum().backward()
    actual.sum().backward()
    torch.testing.assert_close(
        chunked_logits.grad, expected_logits.grad, rtol=1e-12, atol=1e-12
    )


def test_nested_valid_prefixes_restores_variable_lengths_and_gradients():
    input_ids = torch.nested.as_nested_tensor(
        [torch.tensor([11, 12, 13]), torch.tensor([21, 22, 23, 24, 25])],
        layout=torch.jagged,
    )
    padded = torch.arange(12, dtype=torch.float64).reshape(2, 6).requires_grad_(True)

    restored = _nested_valid_prefixes(padded, input_ids)

    assert restored.is_nested
    assert restored.offsets().diff().tolist() == [3, 5]
    first, second = restored.unbind()
    torch.testing.assert_close(first, padded[0, :3])
    torch.testing.assert_close(second, padded[1, :5])

    restored.values().sum().backward()
    expected_grad = torch.tensor(
        [[1, 1, 1, 0, 0, 0], [1, 1, 1, 1, 1, 0]], dtype=torch.float64
    )
    torch.testing.assert_close(padded.grad, expected_grad)


def test_fused_token_output_as_padded_restores_flat_qwen35_output():
    flat = torch.arange(12, dtype=torch.float32)

    padded = _fused_token_output_as_padded(flat, batch_size=2)

    assert padded.shape == (2, 6)
    torch.testing.assert_close(padded[0], flat[:6])
    torch.testing.assert_close(padded[1], flat[6:])
