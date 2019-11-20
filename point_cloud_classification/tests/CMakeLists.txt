set(SOURCE_FILES
    "test.h"
    "test.cpp"
    )

cuda_add_library(tests
    ${SOURCE_FILES}
    OPTIONS -arch=sm_61
    )