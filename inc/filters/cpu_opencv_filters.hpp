#ifndef CUDA_LIBRARY_COMPARISON_CPU_EDGE_DETECTION_HPP
#define CUDA_LIBRARY_COMPARISON_CPU_EDGE_DETECTION_HPP
#include <opencv2/opencv.hpp>

namespace cpu::filters::opencv {

void ApplyLaplacian(const cv::Mat& input_image, cv::Mat& output_image,
                    int kernel_size, double contrast_factor);

}  // namespace cpu::filters::opencv
#endif  // CUDA_LIBRARY_COMPARISON_CPU_EDGE_DETECTION_HPP