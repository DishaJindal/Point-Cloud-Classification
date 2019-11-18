set(SOURCE_FILES
	"layer.h"
	"loss.h"
	"fullyConnectedLayer.h"
	"graphConvolutionLayer.h"
	"globalPoolingLayer.h"
	"RELUActivationLayer.h"
	"sigmoidActivationLayer.h"
	"softmaxActivationLayer.h"
	"crossEntropyLoss.h"
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