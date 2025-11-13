#ifndef CUDA_LIBRARY_COMPARISON_GPU_EDGE_DETECTION_HPP
#define CUDA_LIBRARY_COMPARISON_GPU_EDGE_DETECTION_HPP
#include <opencv2/opencv.hpp>

namespace gpu::filters::opencv {
void ApplyLaplacian(const cv::Mat& input_image, cv::Mat& output_image,
                    int kernel_size, double contrast_factor);

void ApplyLaplacianWithPinnedMem(const cv::Mat& input_image,
                                 cv::Mat& output_image, int kernel_size,
                                 double contrast_factor);
}  // namespace gpu::filters::opencv
#endif  // CUDA_LIBRARY_COMPARISON_GPU_EDGE_DETECTION_HPP