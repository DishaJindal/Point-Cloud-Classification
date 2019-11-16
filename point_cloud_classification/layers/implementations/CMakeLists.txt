set(SOURCE_FILES
	"fullyConnectedLayerCPU.cu"
	"fullyConnectedLayerGPU.cu"
	"globalPoolingCPU.cu"
	"globalPoolingGPU.cu"
	"graphConvolutionLayerCPU.cu"
	"graphConvolutionLayerGPU.cu"
	"RELUActivationLayerCPU.cu"
	"sigmoidActivationLayerCPU.cu"
	"softmaxActivationLayerCPU.cu"
	"RELUActivationLayerGPU.cu"
	"sigmoidActivationLayerGPU.cu"
	"softmaxActivationLayerGPU.cu"
)

cuda_add_library(implementations
    ${SOURCE_FILES}
    OPTIONS -arch=sm_61
)
