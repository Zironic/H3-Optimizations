#pragma once

#if defined(__gfx1200__) || defined(__gfx1201__)
#define COMFY_MMA_GFX12 1
#define COMFY_HAS_WMMA 1
#else
#error "H3 HIP Sparse Kitchen is experimental and supports only gfx1200/gfx1201"
#endif
