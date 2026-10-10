from types import SimpleNamespace
import unittest

from recipes.deepscaler.submission import (
    FINAL_ANSWER_PREFIX, generate_prefilled_submission, prefilled_box_end,
    prefilled_final_record,
)


class CharacterTokenizer:
    eos_token_id = 0

    def encode(self, text, **kwargs):
        return [ord(c) for c in text]

    def decode(self, ids, **kwargs):
        return ''.join(chr(i) for i in ids if i)


class CompletionServer:
    def __init__(self, tokenizer, completion, base_prompt):
        self.tokenizer, self.completion, self.base_prompt = tokenizer, completion, base_prompt
        self.calls = []

    async def generate(self, **kwargs):
        self.calls.append(kwargs)
        offset = len(kwargs['prompt_ids']) - len(self.base_prompt)
        cap = kwargs['sampling_params']['max_tokens']
        ids = self.tokenizer.encode(self.completion[offset:offset + cap])
        return SimpleNamespace(token_ids=ids, log_probs=[-0.25] * len(ids), routed_experts=None)


class SubmissionGenerationTest(unittest.IsolatedAsyncioTestCase):
    async def generate(self, completion, limit=816):
        tokenizer = CharacterTokenizer()
        prompt = tokenizer.encode('problem\nassistant\n' + FINAL_ANSWER_PREFIX)
        server = CompletionServer(tokenizer, completion, prompt)
        output = await generate_prefilled_submission(server, tokenizer, prompt_ids=prompt,
            sampling_params={'temperature': 1, 'top_p': 1, 'top_k': -1}, token_limit=limit)
        return output, server, tokenizer, prompt

    async def test_nested_fraction_stops_before_later_contradicting_answer(self):
        completion = r'\frac{1}{2}} The answer is 9.'
        output, server, tokenizer, prompt = await self.generate(completion)
        kept = r'\frac{1}{2}}'
        self.assertEqual(tokenizer.decode(output.token_ids), kept)
        self.assertEqual(output.token_ids, tokenizer.encode(kept))
        self.assertEqual(output.log_probs, [-0.25] * len(kept))
        self.assertEqual(output.generated_tokens, len(completion))
        self.assertEqual(output.discarded_tokens, len(completion) - len(kept))
        self.assertEqual(server.calls[0]['prompt_ids'], prompt)
        self.assertEqual(prefilled_final_record(kept)['answer'], r'\frac{1}{2}')

    async def test_chunks_continue_exact_tokens_within_the_original_budget(self):
        completion = r'\frac{' + '1' * 70 + r'}{2}} trailing explanation'
        output, server, tokenizer, prompt = await self.generate(completion)
        self.assertEqual(len(server.calls), 2)
        self.assertEqual(server.calls[1]['prompt_ids'], prompt + tokenizer.encode(completion[:64]))
        self.assertEqual(server.calls[1]['sampling_params']['temperature'], 1)
        self.assertLessEqual(output.generated_tokens, 816)
        self.assertEqual(len(output.token_ids), len(output.log_probs))
        self.assertEqual(tokenizer.decode(output.token_ids), completion[:prefilled_box_end(completion)])

    async def test_incomplete_answer_keeps_zero_submission_and_never_exceeds_cap(self):
        output, server, tokenizer, _ = await self.generate('1' * 200, limit=70)
        self.assertEqual([c['sampling_params']['max_tokens'] for c in server.calls], [64, 6])
        self.assertEqual(output.generated_tokens, 70)
        self.assertEqual(len(output.token_ids), 70)
        self.assertIsNone(prefilled_final_record(tokenizer.decode(output.token_ids)))

    def test_escaped_set_braces_and_unclosed_outer_box(self):
        self.assertEqual(prefilled_final_record(r'\{1,2\}}')['answer'], r'\{1,2\}')
        self.assertIsNone(prefilled_final_record(r'\frac{1}{2}'))
        self.assertIsNone(prefilled_final_record('}'))
        self.assertIsNone(prefilled_final_record('...}'))
        self.assertIsNone(prefilled_final_record('<think>5}'))

    async def test_restarted_box_is_submission_without_artificial_outer_close(self):
        for completion in (r'I have finished. \boxed{15}', r'Result: \boxed{15}' + '\n}'):
            with self.subTest(completion=completion):
                output, _, tokenizer, _ = await self.generate(completion)
                kept = tokenizer.decode(output.token_ids)
                self.assertEqual(prefilled_final_record(kept)['answer'], '15')
                self.assertEqual(prefilled_final_record(completion)['answer'], '15')
                self.assertTrue(kept.endswith(r'\boxed{15}'))
                self.assertEqual(len(output.token_ids), len(output.log_probs))

    async def test_quoted_boxes_thinking_and_placeholders_are_not_submissions(self):
        for completion in (r'Use `\boxed{5}` when ready', r'Use `\boxed{5}',
                           r'<think>\boxed{5}</think> Still thinking',
                           r'\boxed{...}', r'\boxed{answer}', r'\boxed{5'):
            with self.subTest(completion=completion):
                self.assertIsNone(prefilled_final_record(completion))
        completion = r'Example `\boxed{99}`. Actual result: \boxed{5}'
        output, _, tokenizer, _ = await self.generate(completion)
        self.assertEqual(tokenizer.decode(output.token_ids), completion)
        self.assertEqual(prefilled_final_record(completion)['answer'], '5')

    async def test_first_complete_submission_is_reference_independent(self):
        completion = r'\boxed{6} then \boxed{5}'
        output, _, tokenizer, _ = await self.generate(completion)
        self.assertEqual(tokenizer.decode(output.token_ids), r'\boxed{6}')
        self.assertEqual(prefilled_final_record(completion)['answer'], '6')
