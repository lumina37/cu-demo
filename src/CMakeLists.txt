function(cud_common_lib_setting name)
    get_target_property(__TYPE ${name} TYPE)
    if (${__TYPE} STREQUAL "INTERFACE_LIBRARY")
        set(__PUB_DEP_SCOPE "INTERFACE")
    else ()
        set(__PUB_DEP_SCOPE "PUBLIC")
    endif ()

    target_compile_features(${name} ${__PUB_DEP_SCOPE} cuda_std_20)

    target_include_directories(${name} ${__PUB_DEP_SCOPE}
            $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
            $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
    )
    target_link_libraries(${name} ${__PUB_DEP_SCOPE}
            CUDA::cublas
            CUDA::cublasLt
    )

    if (CUD_VERBOSE_WARNING)
        if (MSVC)
            target_compile_options(${name} ${__PUB_DEP_SCOPE} /W4)
        else ()
            target_compile_options(${name} ${__PUB_DEP_SCOPE} -Wall -Wextra)
        endif ()
    endif ()
endfunction()

file(GLOB_RECURSE _CUD_SRCS "include/*.cpp")

add_library(cud-lib-static STATIC ${_CUD_SRCS})
cud_common_lib_setting(cud-lib-static)
add_library(cud::lib::static ALIAS cud-lib-static)

add_library(cud-lib-header-only INTERFACE)
cud_common_lib_setting(cud-lib-header-only)
target_compile_definitions(cud-lib-header-only INTERFACE "_CUD_LIB_HEADER_ONLY")
add_library(cud::lib::header-only ALIAS cud-lib-header-only)
