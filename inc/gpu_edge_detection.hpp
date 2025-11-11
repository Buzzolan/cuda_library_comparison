#ifndef CUDA_LIBRARY_COMPARISON_GPU_EDGE_DETECTION_HPP
#define CUDA_LIBRARY_COMPARISON_GPU_EDGE_DETECTION_HPP
#include <opencv2/opencv.hpp>

namespace gpu_edge_detection {
void OpencvGpuLaplacian(const cv::Mat& input_image, cv::Mat& output_image,
                        int kernel_size, double contrast_factor);

void OpencvGpuLaplacian_PinnedMem(const cv::Mat& input_image,
                                  cv::Mat& output_image, int kernel_size,
                                  double contrast_factor);
}  // namespace gpu_edge_detection
#endif  // CUDA_LIBRARY_COMPARISON_GPU_EDGE_DETECTION_HPP