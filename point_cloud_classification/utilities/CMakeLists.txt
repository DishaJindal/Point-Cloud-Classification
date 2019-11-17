set(SOURCE_FILES
    "kernels.h"
    "kernels.cu"
    "matrix.h"
    "matrixCPU.cu"
    "matrixGPU.cu"
    )

cuda_add_library(utilities
    ${SOURCE_FILES}
    OPTIONS -arch=sm_61
    )