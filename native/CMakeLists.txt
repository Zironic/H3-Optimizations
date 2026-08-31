# SPDX-License-Identifier: Apache-2.0
#
# A plain shared library, not a Python extension. The whole point of the C ABI
# in h3_int8_attention_api.cu is that Python reaches it through ctypes, so
# nothing here links against Python and one binary serves every interpreter.
#
# CUTLASS is vendored as headers under native/third_party. The resulting shared
# library still has no external runtime dependency beyond the CUDA toolkit.

cmake_minimum_required(VERSION 3.26)
project(h3_int8_attention LANGUAGES CXX CUDA)

if(MSVC AND CMAKE_GENERATOR MATCHES "^Visual Studio")
    message(FATAL_ERROR
        "The Visual Studio CUDA targets hardcode /EHsc after project flags. "
        "Configure from an x64 Visual Studio developer shell with -G Ninja "
        "so the required /EHs exception boundary is preserved."
    )
endif()

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

if(NOT DEFINED H3_CUDA_ARCHS)
    # Real SASS for every shipped target plus one compute_89 PTX fallback.
    # A compute_120f payload is only forward-compatible within the 12.x
    # family, so carrying it in addition to compute_89 adds size without being
    # the general fallback for a future architecture. Windows toolkits often
    # lack 90a.
    #
    # Turing composes the Ampere-shaped INT8 fragments from m8n8k16 and uses
    # synchronous shared-memory copies in place of cp.async. Runtime use stays
    # fail-closed behind the same per-GPU numerical self-test as newer parts.
    if(WIN32)
        set(H3_CUDA_ARCHS "75-real;80-real;89-real;120f-real;89-virtual")
    else()
        set(H3_CUDA_ARCHS "75-real;80-real;89-real;90a-real;120f-real;89-virtual")
    endif()
endif()
message(STATUS "H3 INT8 attention architectures: ${H3_CUDA_ARCHS}")
set(CMAKE_CUDA_ARCHITECTURES ${H3_CUDA_ARCHS})

add_library(h3_int8_attention SHARED
    src/h3_int8_attention_api.cu
    src/convrot_quantize.cu
    src/h3_fused_q_cutlass.cu
    src/sage_attention/sage_attn_launcher.cu
    src/sage_attention/sage_attn_sparse_launcher.cu
    src/sage_attention/quant_qk_int8.cu
    src/sage_attention/quant_v_int8.cu
    src/sage_attention/v_staging.cu
)

target_include_directories(h3_int8_attention PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/src
    ${CMAKE_CURRENT_SOURCE_DIR}/src/sage_attention
    ${CMAKE_CURRENT_SOURCE_DIR}/third_party/cutlass/include
)

target_compile_options(h3_int8_attention PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:-O3>
    $<$<COMPILE_LANGUAGE:CUDA>:--use_fast_math>
    $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>
    $<$<COMPILE_LANGUAGE:CUDA>:--expt-extended-lambda>
    $<$<COMPILE_LANGUAGE:CUDA>:-U__CUDA_NO_HALF_OPERATORS__>
    $<$<COMPILE_LANGUAGE:CUDA>:-U__CUDA_NO_HALF_CONVERSIONS__>
    $<$<COMPILE_LANGUAGE:CUDA>:-U__CUDA_NO_BFLOAT16_OPERATORS__>
    $<$<COMPILE_LANGUAGE:CUDA>:-U__CUDA_NO_BFLOAT16_CONVERSIONS__>
    $<$<COMPILE_LANGUAGE:CUDA>:-U__CUDA_NO_BFLOAT162_OPERATORS__>
    $<$<COMPILE_LANGUAGE:CUDA>:-U__CUDA_NO_BFLOAT162_CONVERSIONS__>
    $<$<COMPILE_LANGUAGE:CUDA>:--diag-suppress=177>
)

# CUDA 13 compresses fatbins by default, but its size mode is materially
# smaller than the balanced default. Keep older toolkits buildable: they do
# not recognize this option.
if(CMAKE_CUDA_COMPILER_VERSION VERSION_GREATER_EQUAL 13.0)
    target_compile_options(h3_int8_attention PRIVATE
        $<$<COMPILE_LANGUAGE:CUDA>:--compress-mode=size>
    )
endif()

# /EHs, deliberately NOT /EHsc. The trailing c in /EHsc means "extern \"C\"
# functions never throw", and the vendored launchers throw across exactly that
# boundary -- which is what MSVC keeps warning about as C4297. Compiled with
# /EHsc the compiler elides the unwind and the throw terminates the process
# instead of reaching the catch in h3_int8_attention_api.cu, so the guard that
# exists to keep exceptions away from ctypes is itself compiled away.
if(MSVC)
    target_compile_options(h3_int8_attention PRIVATE
        $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=/EHs>
        $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=/Zc:preprocessor>
        $<$<COMPILE_LANGUAGE:CXX>:/EHs>
    )
    # CMake puts /EHsc in both host and CUDA-host defaults. Strip both so a
    # trailing CUDA flag cannot override the required /EHs above.
    string(REPLACE "/EHsc" "" CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}")
    string(REPLACE "/EHsc" "" CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS}")
else()
    # Give the ELF linker independent sections to discard and hide every
    # host symbol except the H3_API functions explicitly marked default in
    # h3_int8_attention_api.cu.
    target_compile_options(h3_int8_attention PRIVATE
        $<$<COMPILE_LANGUAGE:CXX>:-ffunction-sections>
        $<$<COMPILE_LANGUAGE:CXX>:-fdata-sections>
        $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=-ffunction-sections,-fdata-sections>
    )
    target_link_options(h3_int8_attention PRIVATE LINKER:--gc-sections)
    set_target_properties(h3_int8_attention PROPERTIES
        CXX_VISIBILITY_PRESET hidden
        CUDA_VISIBILITY_PRESET hidden
        VISIBILITY_INLINES_HIDDEN YES
    )
endif()

# Static CUDA runtime: one fewer thing for a user to have on their path.
set_target_properties(h3_int8_attention PROPERTIES
    CUDA_RUNTIME_LIBRARY Static
    CUDA_SEPARABLE_COMPILATION OFF
    POSITION_INDEPENDENT_CODE ON
)

if(WIN32)
    set_target_properties(h3_int8_attention PROPERTIES PREFIX "")
endif()

install(TARGETS h3_int8_attention LIBRARY DESTINATION . RUNTIME DESTINATION .)
