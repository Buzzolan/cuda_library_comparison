#ifndef CUDA_LIBRARY_COMPARISON_CPU_EDGE_DETECTION_HPP
#define CUDA_LIBRARY_COMPARISON_CPU_EDGE_DETECTION_HPP
#include <opencv2/opencv.hpp>

namespace cpu_edge_detection {

void OpencvLaplacian(const cv::Mat& input_image, cv::Mat& output_image,
                     int kernel_size, double contrast_factor);

}  // namespace cpu_edge_detection
#endif  // CUDA_LIBRARY_COMPARISON_CPU_EDGE_DETECTION_HPP