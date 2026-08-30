'''Validated configuration for bounded H3 MLP activation execution.'''

from dataclasses import dataclass

MODE_BF16 = 'mlp_chunked_bf16'
MODE_NATIVE = 'mlp_chunked_native'
MODE_FP8 = 'mlp_chunked_fp8'
MODE_CONVROT_INT8_RUNTIME = 'mlp_chunked_convrot_int8_runtime'
MODE_CONVROT_2SLICE = 'mlp_chunked_convrot_2slice'
IMPLEMENTED_MODES = (
    MODE_BF16,
    MODE_NATIVE,
    MODE_FP8,
    MODE_CONVROT_INT8_RUNTIME,
    MODE_CONVROT_2SLICE,
)
DEFAULT_MODE = MODE_NATIVE

DEFAULT_CHUNK_ROWS = 4_096
DEFAULT_ALIGNMENT = 256


@dataclass(frozen=True)
class ActivationMemoryConfig:
    mode: str = DEFAULT_MODE
    chunk_rows: int = DEFAULT_CHUNK_ROWS
    alignment: int = DEFAULT_ALIGNMENT
    strict: bool = True
    prefer_held_weights: bool = True

    def __post_init__(self):
        if self.mode not in IMPLEMENTED_MODES:
            raise ValueError(
                'MLP memory mode %r is unavailable; implemented modes: %s'
                % (self.mode, ', '.join(IMPLEMENTED_MODES))
            )
        chunk_rows = int(self.chunk_rows)
        if (
            isinstance(self.chunk_rows, bool)
            or chunk_rows != self.chunk_rows
            or chunk_rows <= 0
        ):
            raise ValueError('chunk_rows must be a positive integer')
        if int(self.alignment) <= 0:
            raise ValueError('alignment must be positive')

    @property
    def native_swiglu(self):
        return self.mode == MODE_NATIVE

    @property
    def bf16_swiglu(self):
        return self.mode == MODE_BF16

    @property
    def fp8(self):
        return self.mode == MODE_FP8

    @property
    def convrot_2slice(self):
        return self.mode == MODE_CONVROT_2SLICE

    @property
    def runtime_convrot_int8(self):
        return self.mode == MODE_CONVROT_INT8_RUNTIME

    @property
    def signature(self):
        return (
            self.mode,
            int(self.chunk_rows),
            int(self.alignment),
            bool(self.strict),
            bool(self.prefer_held_weights),
        )
