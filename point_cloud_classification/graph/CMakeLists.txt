set(SOURCE_FILES
    "graph.h"
    "graph.cpp"
    "graphKD.cpp"
    "graphGPU.cu"
    "nanoflann.hpp"
    )

cuda_add_library(graph
    ${SOURCE_FILES}
    OPTIONS -arch=sm_61
    )