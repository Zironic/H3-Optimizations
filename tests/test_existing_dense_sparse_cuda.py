'''CUDA contract for unknown dense-consumer fail-open behavior.

This does not validate ROCm execution. It validates the architecture-neutral
adapter behavior with real GPU tensors: a dense consumer can pass the startup
geometry probe, reject a later packed sparse shape, and the adapter must recover
by executing the original full dense problem.
'''

from pathlib import Path
import sys
import unittest
from unittest import mock

import torch

PACK = Path(__file__).resolve().parents[1]
ROOT = next(parent for parent in PACK.parents if (parent / 'comfy').is_dir())
sys.path.insert(0, str(PACK))
sys.path.insert(0, str(ROOT))

from h3_optimizations.attention.sparse import existing_dense_sparse  # noqa: E402


requires_cuda = unittest.skipUnless(
    torch.cuda.is_available() and not getattr(torch.version, 'hip', None),
    'requires NVIDIA CUDA',
)


@requires_cuda
class ExistingDenseSparseCUDAFailOpenTests(unittest.TestCase):
    def setUp(self):
        existing_dense_sparse._probe_results.clear()
        existing_dense_sparse._runtime_fallback_warned = False

    def test_probe_passes_then_unseen_packed_shape_falls_back_to_full_dense(self):
        calls = []

        def unknown_dense(
            q,
            k,
            v,
            heads,
            mask=None,
            skip_reshape=True,
            skip_output_reshape=True,
            transformer_options=None,
        ):
            del mask, skip_reshape, skip_output_reshape, transformer_options
            calls.append((int(q.shape[-2]), int(k.shape[-2]), int(q.shape[0]), int(heads)))
            # Force the probe to select 128Q x 128KV. The 128 geometry probe
            # exercises 128/256/249 KV rows, but not the 384-row packed problem
            # used below. This simulates an otherwise-compatible unknown dense
            # backend with a runtime shape restriction the probe did not cover.
            if int(q.shape[-2]) == 64:
                raise RuntimeError('unknown consumer rejects 64Q')
            if int(k.shape[-2]) == 384:
                raise RuntimeError('unknown consumer rejects 384KV packed shape')
            return existing_dense_sparse._reference_attention(q, k, v)

        with mock.patch.object(
            existing_dense_sparse.h3_model,
            'optimized_attention',
            side_effect=unknown_dense,
        ):
            spec = existing_dense_sparse.probe_existing_dense_sparse(
                device='cuda',
                force=True,
            )
            self.assertEqual((spec.q_tile, spec.kv_tile), (128, 128))

            generator = torch.Generator(device='cuda').manual_seed(20260831)
            shape = (1, 1, 512, existing_dense_sparse.HEAD_DIM)
            q = torch.randn(shape, dtype=torch.bfloat16, device='cuda', generator=generator)
            k = torch.randn(shape, dtype=torch.bfloat16, device='cuda', generator=generator)
            v = torch.randn(shape, dtype=torch.bfloat16, device='cuda', generator=generator)

            # Delta route [0, 1, 1] -> absolute tiles [0, 1, 2]. Each 128Q
            # entry therefore presents 384 KV rows to the dense consumer.
            lut = torch.tensor(
                [[[[0, 1, 1], [0, 1, 1], [0, 1, 1], [0, 1, 1]]]],
                dtype=torch.int32,
                device='cuda',
            )
            prepared = existing_dense_sparse.PreparedExistingDenseSparse(
                q=q,
                k=k,
                v=v,
                lut=lut,
                valid=torch.full((1, 1, 4), 3, dtype=torch.int32, device='cuda'),
                metadata={
                    'dense_q_tiles': 0,
                    'sparse_q_tiles': 4,
                    'kv_tiles': 4,
                    'pure_video_kv_tiles': 4,
                    'retained_video_kv_tiles': 3,
                },
                transformer_options={},
            )
            backend = existing_dense_sparse.ExistingDenseSparseBackend(spec=spec)
            actual = backend.execute(prepared)
            expected = existing_dense_sparse._reference_attention(q, k, v)

        torch.cuda.synchronize()
        self.assertTrue(torch.equal(actual, expected))
        self.assertIn((128, 384, 4, 1), calls)
        self.assertIn((512, 512, 1, 1), calls)


if __name__ == '__main__':
    unittest.main()
