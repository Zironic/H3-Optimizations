'''Public ComfyUI node registration for H3 Optimizations.'''

from comfy_api.latest import ComfyExtension

# Import first so every public node, including Sparse Attention used without the
# Memory node, resolves QKV through the same streamed-BF16 priority policy.
from . import apply_policy as _apply_policy  # noqa: F401
# Experimental AMD policy is deliberately layered after the ordinary policy so
# it only changes the RDNA2 Auto fallback and leaves every other resolver path.
from . import amd_policy as _amd_policy  # noqa: F401
from .aimdo_limiter import H3AIMDOResidencyLimiter
from .memory_migration_node import H3MemoryOptimization
from .nodes import (
    H3SparseAttention,
    H3SparseAttentionAdvanced,
)


class H3OptimizationsExtension(ComfyExtension):
    '''Register the production H3 optimization nodes.'''

    async def get_node_list(self):
        return [
            H3MemoryOptimization,
            H3AIMDOResidencyLimiter,
            H3SparseAttention,
            H3SparseAttentionAdvanced,
        ]
