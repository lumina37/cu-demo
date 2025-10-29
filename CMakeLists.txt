cmake_minimum_required(VERSION 3.25 FATAL_ERROR)

project(CUDADemo
        VERSION 0.0.1
        DESCRIPTION "CUDA demo"
        LANGUAGES CUDA CXX
)

# Options
option(CUD_ENABLE_TESTS "Enable tests" OFF)
option(CUD_VERBOSE_WARNING "Show verbose compiler warnings" OFF)

# Dependencies
include(FetchContent)

find_package(CUDAToolkit)

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
