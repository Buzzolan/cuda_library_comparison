#ifndef CUDA_LIBRARY_COMPARISON_GPU_NPP_EDGE_DETECTION_HPP
#define CUDA_LIBRARY_COMPARISON_GPU_NPP_EDGE_DETECTION_HPP

#include <npp.h>

#include <opencv2/opencv.hpp>

namespace gpu::filters::npp {
void Laplacian_NPP_Device(Npp8u* d_src, int step, NppiSize roi, Npp8u* d_dst);
void Laplacian_NPP_Host(const cv::Mat& image, cv::Mat& out_cpu_img);
void GaussinanFilter_NPP_Device(Npp8u* d_src, int step, NppiSize roi,
                                Npp8u* d_dst);
void GaussianFilter_NPP_Host(const cv::Mat& image, Npp8u* d_dst);
}  // namespace gpu::filters::npp
#endif  // CUDA_LIBRARY_COMPARISON_GPU_NPP_EDGE_DETECTION_HPP