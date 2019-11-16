set(SOURCE_FILES
	"fullyConnectedLayerCPU.cu"
	"fullyConnectedLayerGPU.cu"
"globalPoolingCPU.cu"
	"globalPoolingGPU.cu"
"graphConvolutionLayerCPU.cu"
	"graphConvolutionLayerGPU.cu"
)

cuda_add_library(implementations
    ${SOURCE_FILES}
    OPTIONS -arch=sm_61
)
