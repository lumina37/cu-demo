cmake_minimum_required(VERSION 3.25 FATAL_ERROR)

cmake_policy(SET CMP0104 NEW)
set(CMAKE_OBJECT_PATH_MAX 1024)

project(CUDADemo
        VERSION 0.0.1
        DESCRIPTION "CUDA demo"
        LANGUAGES CUDA CXX
)

# Options
option(CUD_ENABLE_TESTS "Enable tests" OFF)

# Dependencies
include(FetchContent)

set(CUTLASS_ENABLE_CUBLAS ON CACHE BOOL "" FORCE)
set(CUTLASS_NVCC_ARCHS ${CMAKE_CUDA_ARCHITECTURES} CACHE STRING "" FORCE)
FetchContent_Declare(
        cutlass
        GIT_REPOSITORY https://github.com/NVIDIA/cutlass.git
        GIT_TAG v4.2.1
)
FetchContent_MakeAvailable(cutlass)

if (CUD_ENABLE_TESTS)
    include(CTest)

    FetchContent_Declare(
            Catch2
            GIT_REPOSITORY https://github.com/catchorg/Catch2.git
            GIT_TAG v3.8.0
    )
    FetchContent_MakeAvailable(Catch2)
    include(Catch)
endif ()

# Sources
add_subdirectory(src)
add_subdirectory(samples)

if (CUD_ENABLE_TESTS)
    add_subdirectory(tests)
endif ()
