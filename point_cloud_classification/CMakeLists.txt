set(SOURCE_FILES
    "common.h"
    "common.cu"
    "network.h"
    "networkCPU.cu"
    "networkGPU.cu"
    )

include_directories(.)
add_subdirectory(utilities)
add_subdirectory(graph)
add_subdirectory(hidden_layers)
add_subdirectory(tests)

cuda_add_library(point_cloud_classification
    ${SOURCE_FILES}
    OPTIONS -arch=sm_61
    )

target_link_libraries(point_cloud_classification
	utilities
	graph
	hidden_layers
	tests
	)