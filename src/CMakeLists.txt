function(cud_common_lib_setting name)
    target_compile_features(${name} PUBLIC cuda_std_20)

    target_include_directories(${name} PUBLIC
            $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
            $<BUILD_INTERFACE:${cutlass_SOURCE_DIR}/include>
            $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
    )
    target_link_libraries(${name} PUBLIC
            nvidia::cublasLt
            nvidia::cublas
            nvidia::cutlass::cutlass
            nvidia::cutlass::tools::util
    )
endfunction()

file(GLOB_RECURSE _CUD_SRCS "include/*.cpp")

add_library(cud-lib-static STATIC ${_CUD_SRCS})
cud_common_lib_setting(cud-lib-static)
add_library(cud::lib::static ALIAS cud-lib-static)
