set(SOURCE_FILES
    "graph.h"
    "graph.cpp"
    )

cuda_add_library(graph
    ${SOURCE_FILES}
    OPTIONS -arch=sm_61
    )