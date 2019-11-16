set(SOURCE_FILES
    "kernels.h"
    "kernels.cu"
    "matrix.h"
    )

cuda_add_library(utilities
    ${SOURCE_FILES}
    OPTIONS -arch=sm_61
    )