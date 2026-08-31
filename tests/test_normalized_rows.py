'''CPU contracts for lazy pre-attention normalization rows.'''

from pathlib import Path
import unittest

import torch

from h3_optimizations.normalized_rows import (
    NormalizedRows,
    NormalizedRowsUnsupported,
    attention_output_buffer,
)


PACK = Path(__file__).resolve().parents[1]
SOURCE = PACK / 'h3_optimizations'

LAZY_CONSUMER_TESTS = {
    'attention_forward.py': (
        'test_bf16_qkv.py',
        'test_dense_streaming_keeps_lazy_input_separate_from_output',
    ),
    'dense_streamed_sage.py': (
        'test_dense_streamed_sage.py',
        'test_lazy_normalized_rows_remain_the_streamed_q_source',
    ),
    'kitchen_qkv.py': (
        'test_streamed_kitchen_output.py',
        'test_dense_kitchen_streamed_q_keeps_lazy_input_separate_from_output',
    ),
    'attention/sparse/existing_dense_sparse.py': (
        'test_rocm_existing_dense_failopen.py',
        'test_streamed_existing_dense_keeps_lazy_input_separate_from_output',
    ),
    'attention/sparse/frost_bf16_streamed.py': (
        'test_streamed_frost_bf16.py',
        'test_execute_keeps_lazy_input_separate_from_attention_output',
    ),
    'attention/sparse/existing_dense_sparse.py': (
        'test_existing_dense_sparse.py',
        'test_streamed_execute_keeps_lazy_input_separate_from_output',
    ),
    'attention/sparse/kitchen_sparse.py': (
        'test_streamed_kitchen_output.py',
        'test_sparse_kitchen_keeps_lazy_residual_separate_from_output',
    ),
    'attention/sparse/kitchen_streamed_q.py': (
        'test_streamed_kitchen_output.py',
        'test_sparse_kitchen_streamed_q_keeps_lazy_input_separate_from_output',
    ),
    'attention/sparse/sparse_sage_streamed.py': (
        'test_streamed_sparse_sage.py',
        'test_execute_keeps_lazy_input_separate_from_attention_output',
    ),
    'attention/sparse/triton_bf16_streamed.py': (
        'test_streamed_triton_bf16.py',
        'test_execute_keeps_lazy_input_separate_from_attention_output',
    ),
}


def _apply_modulation(rows, shift, scale, selector):
    rows.mul_(1.0 + scale[selector].to(rows.dtype))
    rows.add_(shift[selector].to(rows.dtype))


class NormalizedRowsTests(unittest.TestCase):
    @staticmethod
    def _case():
        x = torch.arange(80, dtype=torch.float32).reshape(10, 8) / 20
        shift = torch.arange(32, dtype=torch.float32).reshape(4, 8) / 100
        scale = torch.arange(32, dtype=torch.float32).reshape(4, 8) / 200
        selector = torch.tensor([3, 0, 2, 1, 3], dtype=torch.long)
        segments = ((0, 2, 1), (2, 7, selector), (7, 10, 2))
        source = NormalizedRows(
            x,
            lambda rows: rows * 0.5,
            segments,
            shift,
            scale,
            _apply_modulation,
        )

        expected = x * 0.5
        _apply_modulation(expected[0:2], shift, scale, 1)
        _apply_modulation(expected[2:7], shift, scale, selector)
        _apply_modulation(expected[7:10], shift, scale, 2)
        return x, source, expected

    def test_contiguous_slices_and_materialization_match_full_math(self):
        _x, source, expected = self._case()
        self.assertTrue(torch.equal(source[1:9], expected[1:9]))
        self.assertTrue(torch.equal(source.materialize(), expected))

    def test_unsorted_row_gather_matches_full_math(self):
        _x, source, expected = self._case()
        index = torch.tensor([8, 2, 6, 0, 4], dtype=torch.long)
        self.assertTrue(
            torch.equal(
                source.index_select(0, index),
                expected.index_select(0, index),
            )
        )

    def test_attention_output_is_distinct_and_allocated_once(self):
        x, source, expected = self._case()
        output = attention_output_buffer(source)
        self.assertIs(output, attention_output_buffer(source))
        self.assertIsNot(output, x)
        output.copy_(expected + 5)
        self.assertTrue(torch.equal(output, expected + 5))
        self.assertTrue(torch.equal(source.materialize(), expected))
        self.assertIs(attention_output_buffer(x), x)

    def test_unsupported_tensor_operations_request_materialization(self):
        _x, source, _expected = self._case()
        with self.assertRaises(NormalizedRowsUnsupported):
            torch.mean(source)
        with self.assertRaises(NormalizedRowsUnsupported):
            source.reshape(5, 16)
        with self.assertRaises(NormalizedRowsUnsupported):
            source.index_select(1, torch.tensor([0], dtype=torch.long))

    def test_every_output_buffer_consumer_has_a_lazy_source_regression(self):
        actual = set()
        for path in SOURCE.rglob('*.py'):
            if path.name == 'normalized_rows.py':
                continue
            if 'attention_output_buffer(' in path.read_text(encoding='utf-8'):
                actual.add(path.relative_to(SOURCE).as_posix())

        self.assertEqual(actual, set(LAZY_CONSUMER_TESTS))
        for relative, (test_file, test_name) in LAZY_CONSUMER_TESTS.items():
            with self.subTest(consumer=relative):
                test_source = (PACK / 'tests' / test_file).read_text(
                    encoding='utf-8'
                )
                self.assertIn('def %s(' % test_name, test_source)


if __name__ == '__main__':
    unittest.main()
