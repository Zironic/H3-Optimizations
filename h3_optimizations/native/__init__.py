"""The pack's own compiled INT8 attention, loaded through ctypes.

Deliberately name-for-name compatible with the comfy_kitchen surface this pack
already calls, so ``kitchen_qkv.py`` and the sparse backend can hold either
module without branching on which one they got. That compatibility is the
point: the integration was written months ago against a Kitchen API that was
never released, and this makes it run without changing the integration.
"""

from .loader import (
    ABI_VERSION,
    NativeCallError,
    NativeUnavailableError,
    check,
    is_available,
    load,
    route_encoding,
    unavailable_reason,
)
from .int8_attention import (
    BlockSparseRoute,
    PrequantizedInt8Attention,
    SPARSE_GEOMETRIES,
    block_sparse_int8_attention_from_prequantized,
    block_sparse_int8_attention_with_lse_from_prequantized,
    int8_attention_from_prequantized,
    int8_attention_is_available,
    prequantize_int8_attention,
)
from .convrot import (
    int8_rowwise_convrot256_is_available,
    quantize_int8_rowwise_convrot256,
)
from .fused_q import fused_h3_q_from_int8, fused_h3_q_is_available
from .producer import (
    INT8_ATTENTION_PRODUCER_ABI_VERSION,
    SUPPORTS_STRIDED_QK_CHUNK,
    Int8AttentionKAnchor,
    Int8AttentionProducer,
    Int8AttentionProducerSpec,
    Int8AttentionProducerUnavailableError,
    create_int8_attention_producer,
    finalize_int8_attention_producer,
    int8_attention_k_anchor_positions,
    int8_attention_producer_is_available,
    int8_attention_producer_spec,
    quantize_int8_attention_k_chunk,
    quantize_int8_attention_q,
    quantize_int8_attention_q_chunk,
    quantize_int8_attention_qk_chunk,
    quantize_int8_attention_v,
    select_int8_attention_k_anchor,
)

__all__ = [
    'SUPPORTS_STRIDED_QK_CHUNK',
    'ABI_VERSION',
    'BlockSparseRoute',
    'INT8_ATTENTION_PRODUCER_ABI_VERSION',
    'Int8AttentionKAnchor',
    'Int8AttentionProducer',
    'Int8AttentionProducerSpec',
    'Int8AttentionProducerUnavailableError',
    'NativeCallError',
    'NativeUnavailableError',
    'PrequantizedInt8Attention',
    'SPARSE_GEOMETRIES',
    'block_sparse_int8_attention_from_prequantized',
    'block_sparse_int8_attention_with_lse_from_prequantized',
    'check',
    'create_int8_attention_producer',
    'finalize_int8_attention_producer',
    'fused_h3_q_from_int8',
    'fused_h3_q_is_available',
    'int8_attention_from_prequantized',
    'int8_attention_is_available',
    'int8_attention_k_anchor_positions',
    'int8_attention_producer_is_available',
    'int8_attention_producer_spec',
    'int8_rowwise_convrot256_is_available',
    'is_available',
    'load',
    'prequantize_int8_attention',
    'quantize_int8_rowwise_convrot256',
    'quantize_int8_attention_k_chunk',
    'quantize_int8_attention_q',
    'quantize_int8_attention_q_chunk',
    'quantize_int8_attention_qk_chunk',
    'quantize_int8_attention_v',
    'route_encoding',
    'select_int8_attention_k_anchor',
    'unavailable_reason',
]
