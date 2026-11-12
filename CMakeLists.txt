cmake_minimum_required(VERSION 3.24)
project(character_transformer LANGUAGES CXX CUDA)

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build configuration" FORCE)
endif()

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

option(ATTENTION_DEBUG_DUMPS "Enable existing DEBUG tensor dumps and one-epoch run" OFF)
option(ATTENTION_ENGINE_RUN "Enable existing ENGINE_RUN code paths" OFF)

# main.cpp includes CUDA-specific headers. Compile it with nvcc, as in the
# original command; each translation unit gets its own cached object file.
set_source_files_properties(src/main.cpp src/include/utils.cpp PROPERTIES LANGUAGE CUDA)

add_executable(attention
    src/main.cpp
    src/include/helper.cpp
    src/include/utils.cpp
    src/kernel/utils.cu
    src/forward/Kernel/layer_norm.cu
    src/forward/Kernel/embedding.cu
    src/forward/Kernel/linear.cu
    src/forward/Kernel/attention_head.cu
    src/forward/Kernel/interface.cu
    src/backpropagation/Kernel/interface_back.cu
    src/backpropagation/Kernel/flash_attention.cu
    src/optimizer/Kernel/AdamW.cu
)

set_target_properties(attention PROPERTIES
    CXX_STANDARD 17
    CXX_STANDARD_REQUIRED YES
    CUDA_STANDARD 17
    CUDA_STANDARD_REQUIRED YES
    # Kernels and their launch wrappers reside in the same translation unit.
    # No extra device-link stage is necessary.
    CUDA_SEPARABLE_COMPILATION OFF
    RUNTIME_OUTPUT_DIRECTORY "${PROJECT_SOURCE_DIR}/src/bin"
)

target_compile_definitions(attention PRIVATE
    $<$<BOOL:${ATTENTION_DEBUG_DUMPS}>:DEBUG>
    $<$<BOOL:${ATTENTION_ENGINE_RUN}>:ENGINE_RUN>
)

# Resolve the existing relative corpus/cache paths from the repository root.
add_custom_target(run
    COMMAND $<TARGET_FILE:attention>
    DEPENDS attention
    WORKING_DIRECTORY "${PROJECT_SOURCE_DIR}"
    USES_TERMINAL
)
