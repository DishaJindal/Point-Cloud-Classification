set(SOURCE_FILES
	"layer.h"
	"fullyConnectedLayer.h"
	"graphConvolutionLayer.h"
	"globalPoolingLayer.h"
)

include_directories(.)
add_subdirectory(implementations)

cuda_add_library(layers
    ${SOURCE_FILES}
    OPTIONS -arch=sm_61
    )

target_link_libraries(layers
    implementations
    )