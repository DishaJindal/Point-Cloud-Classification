# Point-Cloud-Classification
**Team**: [Saket Karve](https://www.linkedin.com/in/saket-karve-43930511b/), [Disha Jindal](https://www.linkedin.com/in/disha-jindal/), [Kushagra Goel](https://www.linkedin.com/in/kushagragoel/)

[Proposal](./milestones/Proposal.pdf)
[Data](https://drive.google.com/file/d/1c2gZ2URDaMdimGsyt1sK17F6H6byQCMq/view?usp=sharing)

![](./img/Chair.gif)

# Overview

The aim of our project is 3D point cloud classification using Graph convolutional neural networks on CUDA. GCNs are very effective because they efficiently exploit the local structure in point clouds. The architecture involves nearest neighbor search to construct the graph from the point clouds.  The graph layers take a graph as input and propagate the input preserving the graph structure across layers. Graph neural networks are an ongoing topic of research and have applications in various fields. 

In this project, we have implemented a full end to end graph convolution network on GPU using CUDA and CPU as a benchmark for performance analysis. An optimized version of the kernels for reduction and basic matrix operations is also implemented. 

# Features Implemented

- Designed a framework for implementing any generic neural network architecture
- Farthest sampling of point clouds
- Graph generation of data on CPU and GPU
- CPU implementation of all layers
- GPU implementation of all layers
- Optimization of various kernels using memory optimiztion and streams

# Network Architecture

![](./img/architecture.png)

# Some Predictions

<p align="center"><img src="./img/table.jpg" width="500"/> </p>

The above input was predicted as a 'Table' by our network.

<p align="center"><img src="./img/toilet.jpeg" width="500"/> </p>

The above input was predicted as a 'Toilet' by our network.

# Performance Analysis

![](./img/fwd.png)

![](./img/layer_wise.png)

# References

[A Graph-CNN for 3D Point Cloud Classification](https://arxiv.org/pdf/1812.01711.pdf)

[Towards Efficient Large-Scale Graph Neural Network Computing](https://arxiv.org/pdf/1810.08403.pdf)

[Convolutional Neural Networks on Graphs with Fast Localized Spectral Filtering](https://arxiv.org/pdf/1606.09375.pdf)

[How to do Deep Learning on Graphs with Graph Convolutional Networks](https://towardsdatascience.com/how-to-do-deep-learning-on-graphs-with-graph-convolutional-networks-7d2250723780)

[KroneckerIntro](http://www.mathcs.emory.edu/~nagy/courses/fall10/515/KroneckerIntro.pdf)

# Tasks

| Task                | &#x265a;Saket&#x265a;  | 	&#x265b; Disha 	&#x265b; | &#x2654; Kushagra &#x2654;            |
| ---                 | ---    | ---        | ---                  |
|  RELUActivationLayerGPU    |   |       |              |  
|  crossEntropyLossGPU   |   |       |              |
|  fullyConnectedLayerGPU   |   |       |              |
|  globalPoolingGPU   |   |       |              |
|  graphConvolutionLayerGPU   |   |       |              |
|  sigmoidActivationLayerGPU   |   |       |              |
|  softmaxActivationLayerGPU   |   |       |              |
| Graph Convolution CPU Forward    | &#x2611;   |  |            |
| Droput CPU Forward    | | &#x2611;     |           |
| Dropout CPU Backward    |  | &#x2611;        |            |
| Global Pool CPU Forward    |   | &#x2611;       |            |
| Global Pool CPU Backward    |  | &#x2611;        |             |
| Eigen Math Library    | &#x25CB;  |    |           |
| Data Sampling    | |   | &#x2611;              |
| Data Loading    | |   | &#x2611;              |
| Label Loading    |  |    |  &#x2611;              |
| Architecture     |  |  &#x25CB;  |                |
| Train on CPU     |  |  &#x25CB;  |               |
| GPU kernels for Utils   |   |                | &#x25CB; |
| Layers on GPU   |    |                |    |



|Markdown Icon Legend                 | &#x2611; Completed    | &#x2612; Not Doing      |&#x25CB; Currently Working on            |  
| ---                 | ---    | ---        | ---                  |

