#ifndef CUDA_LIBRARY_COMPARISON_GPU_NPP_EDGE_DETECTION_HPP
#define CUDA_LIBRARY_COMPARISON_GPU_NPP_EDGE_DETECTION_HPP

#include <npp.h>

#include <opencv2/opencv.hpp>

namespace gpu::filters::npp {
void ApplyLaplacianWithGaussian(const cv::Mat& image, cv::Mat& out_cpu_img);
}  // namespace gpu::filters::npp
#endif  // CUDA_LIBRARY_COMPARISON_GPU_NPP_EDGE_DETECTION_HPP