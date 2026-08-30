cmake_minimum_required(VERSION 3.26)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_HIP_STANDARD 20)
set(CMAKE_HIP_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

if(NOT DEFINED H3_HIP_ARCHS)
    set(H3_HIP_ARCHS "gfx1200;gfx1201")
endif()
foreach(arch IN LISTS H3_HIP_ARCHS)
    if(NOT arch STREQUAL "gfx1200" AND NOT arch STREQUAL "gfx1201")
        message(FATAL_ERROR
            "Experimental H3 HIP Sparse Kitchen supports gfx1200/gfx1201; got ${arch}"
        )
    endif()
endforeach()
set(CMAKE_HIP_ARCHITECTURES "${H3_HIP_ARCHS}")

project(h3_hip_sparse_kitchen LANGUAGES CXX HIP)

if(NOT DEFINED HIP_PLATFORM)
    set(HIP_PLATFORM "amd")
endif()
find_package(hip REQUIRED CONFIG)

add_library(h3_hip_sparse_kitchen SHARED
    src/h3_hip_sparse_api.hip
    src/sage_attention/int8_attn_sparse.hip
    src/sage_attention/quant_qk_int8.hip
    src/sage_attention/quant_v_int8.hip
)
target_include_directories(h3_hip_sparse_kitchen PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/src
    ${CMAKE_CURRENT_SOURCE_DIR}/src/sage_attention
)
target_compile_options(h3_hip_sparse_kitchen PRIVATE
    $<$<COMPILE_LANGUAGE:HIP>:-O3>
)
set_source_files_properties(src/sage_attention/int8_attn_sparse.hip PROPERTIES
    COMPILE_OPTIONS "$<$<COMPILE_LANGUAGE:HIP>:-fno-fast-math>"
)
target_link_libraries(h3_hip_sparse_kitchen PRIVATE hip::host)
set_target_properties(h3_hip_sparse_kitchen PROPERTIES
    CXX_VISIBILITY_PRESET hidden
    HIP_VISIBILITY_PRESET hidden
    VISIBILITY_INLINES_HIDDEN YES
)

install(TARGETS h3_hip_sparse_kitchen LIBRARY DESTINATION . RUNTIME DESTINATION .)
