function(cud_add_sample name src)
    add_executable(${name} ${src})

    target_link_libraries(${name} PRIVATE cud::lib::static)

    set_target_properties(${name} PROPERTIES
            CUDA_SEPARABLE_COMPILATION ON
    )
endfunction()

cud_add_sample(cud-bin-sgemm-cublas sgemm/cublas.cu)
cud_add_sample(cud-bin-sgemm-cublaslt sgemm/cublaslt.cu)
cud_add_sample(cud-bin-sgemm-simt-v6-sub-thread-tile sgemm/simt/v6-sub-thread-tile.cu)
