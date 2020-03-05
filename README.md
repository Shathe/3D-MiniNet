
<div align="center">  
  
# 3D-MiniNet: Learning a 2D Representation from Point Clouds for Fast and Efficient 3D LIDAR Semantic Segmentation


[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/3d-mininet-learning-a-2d-representation-from/3d-semantic-segmentation-on-semantickitti)](https://paperswithcode.com/sota/3d-semantic-segmentation-on-semantickitti?p=3d-mininet-learning-a-2d-representation-from)

[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/3d-mininet-learning-a-2d-representation-from/real-time-3d-semantic-segmentation-on)](https://paperswithcode.com/sota/real-time-3d-semantic-segmentation-on?p=3d-mininet-learning-a-2d-representation-from)

[![Paper](https://img.shields.io/badge/Paper-PDF-red)](https://arxiv.org/pdf/2002.10893.pdf)

</div>

## Abstract   
LIDAR semantic segmentation, which assigns a semantic label to each 3D point measured by the LIDAR, is becoming an essential task for many robotic applications such as autonomous driving. Fast and efficient semantic segmentation methods are needed to match the strong computational and temporal restrictions of many of these real-world applications.
This work presents 3D-MiniNet, a novel approach for LIDAR semantic segmentation that combines 3D and 2D learning layers. It first learns a 2D representation from the raw points through a novel projection which extracts local and global information from the 3D data. This representation is fed to an efficient 2D Fully Convolutional Neural Network (FCNN) that produces a 2D semantic segmentation. These 2D semantic labels are re-projected back to the 3D space and enhanced through a post-processing module. The main novelty in our strategy relies on the projection learning module. Our detailed ablation study shows how each component contributes to the final performance of 3D-MiniNet. We validate our approach on well known public benchmarks (SemanticKITTI and KITTI), where 3D-MiniNet gets state-of-the-art results while being faster and more parameter-efficient than previous methods.


 
## Introduction
This repository contains the implementation of **3D-MiniNet**, a fast and efficient method for semantic segmentation of LIDAR point clouds.

The following figure shows the basic building block of our **3D-MiniNet**:

<p align="center"> <img src="figs/3D-MiniNet.png" width="100%"> </p>

3D-MiniNet overview. It takes *P* groups of *N* points each and computes semantic segmentation of the *M* points of the point cloud where *PxN=M*.

It consists of two main modules: our proposed learning module (on the left) which learns a 2D tensor which is fed to the second module, an efficient FCNN backbone (on the right) which computes the 2D semantic segmentation. Each 3D point of the point cloud is given a semantic label based on the 2D segmentation.

## Code will be soon released (Pytorch and Tensorflow implementation)  
The Pytorch implementation will use the [RangeNet++ (Milioto et al. IROS2019) repo](https://github.com/PRBonn/lidar-bonnetal) as its code base.

The Tensorflow implementation will use the [LuNet (Biasutti et al. ICCVW2019) repo](https://github.com/pbias/lunet) as its code base.

The code will be released around April.
In the meantime, check those works out and give them love, they are really good works!



