set(SOURCE_FILES
    "kernels.h"
    "kernels.cu"
    "matrix.h"
    "matrixCPU.cu"
    "matrixGPU.cu"
    "utils.h"
    "parameters.cpp"
    )

cuda_add_library(utilities
    ${SOURCE_FILES}
    OPTIONS -arch=sm_61
    )