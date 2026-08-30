'''Static ownership checks for the standalone repository.'''

from pathlib import Path
import re
import unittest

PACK = Path(__file__).resolve().parents[1]
SOURCE = PACK / 'h3_optimizations'


class SourceIsolationTests(unittest.TestCase):
    def test_python_source_has_no_experimental_pack_dependencies(self):
        banned = (
            'ComfyUI-H3-Extended',
            'h3_attention',
            'h3_activation_memory',
            'h3_runtime',
            'h3_probe',
            'epilogue',
            'minimax_h3::',
            'torch.no_grad',
            'torch.inference_mode',
            'torch.cuda.empty_cache',
            'torch.cuda.synchronize',
        )
        # The ban keeps stalls and allocator churn out of the hot path. The
        # native self-test is the one place a synchronize is the point: it runs
        # once at startup to catch asynchronous kernel faults, which surface
        # only at a sync and would otherwise appear later as an unrelated bug.
        exempt = {
            SOURCE / 'native' / 'selftest.py': ('torch.cuda.synchronize',),
            SOURCE / 'native' / 'hip_selftest.py': ('torch.cuda.synchronize',),
        }
        for path in SOURCE.rglob('*.py'):
            text = path.read_text(encoding='utf-8')
            allowed = exempt.get(path, ())
            for fragment in banned:
                if fragment in allowed:
                    continue
                self.assertNotIn(fragment, text, '%s contains %s' % (path, fragment))

    def test_every_custom_op_uses_the_repo_namespace(self):
        declarations = []
        for path in SOURCE.rglob('*.py'):
            text = path.read_text(encoding='utf-8')
            for match in re.finditer(r'custom_op\(', text):
                declarations.append(text[match.start():match.start() + 200])
        self.assertTrue(declarations)
        for declaration in declarations:
            self.assertIn('h3_optimizations::', declaration)

    def test_package_metadata_points_to_the_canonical_repo(self):
        metadata = (PACK / 'pyproject.toml').read_text(encoding='utf-8')
        self.assertIn(
            'https://github.com/Zironic/H3-Optimizations',
            metadata,
        )

    def test_dense_kitchen_integration_uses_only_public_carrier_apis(self):
        source = (SOURCE / 'kitchen_qkv.py').read_text(encoding='utf-8')
        self.assertNotIn('._C', source)
        self.assertNotIn('PrequantizedInt8Attention(', source)

    def test_dense_sage_stack_stays_at_the_h3_apply_boundary(self):
        production_boundaries = (
            SOURCE / 'attention' / '__init__.py',
            SOURCE / 'qkv' / '__init__.py',
            SOURCE / 'qkv' / 'projectors.py',
        )
        banned = (
            'SM89SageMemoryEfficientBackend',
            'DenseFusedQKVProjector',
            'sage_mem_eff',
            'dense_fused_qkv',
        )
        for path in production_boundaries:
            text = path.read_text(encoding='utf-8')
            for fragment in banned:
                self.assertNotIn(fragment, text, '%s exports %s' % (path, fragment))
        apply_source = (SOURCE / 'apply.py').read_text(encoding='utf-8')
        self.assertIn('StreamedDenseSageBackend', apply_source)
        self.assertIn('StreamedDenseSageQKVProjector', apply_source)
        self.assertNotIn('ProjectedSM89SageBackend', apply_source)
        self.assertNotIn('DenseFusedQKVProjector', apply_source)

    def test_sparse_production_uses_streamed_convrot_projector(self):
        apply_source = (SOURCE / 'apply.py').read_text(encoding='utf-8')
        projector_source = (
            SOURCE / 'qkv' / 'projectors.py'
        ).read_text(encoding='utf-8')
        backend_source = (
            SOURCE / 'attention' / 'sparse' / 'backend.py'
        ).read_text(encoding='utf-8')

        self.assertIn('SparseFusedQKVProjector(', apply_source)
        self.assertIn('chunk_rows=4096', apply_source)
        self.assertIn('StreamedSparseSageQKVProjector', projector_source)
        self.assertNotIn('FusedQKVProjector as Implementation', projector_source)
        # The backend-level fallback remains the old chunked projector for
        # callers that construct fused mode without the provider resolver.
        self.assertIn('ChunkedSparseQKVProjector(kernel_spec)', backend_source)
        self.assertNotIn('FusedQKVProjector()', backend_source)


if __name__ == '__main__':
    unittest.main()
