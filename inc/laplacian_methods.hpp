#pragma once
#include <npp.h>  // Add this include for Npp8u definition

#include <opencv2/opencv.hpp>

cv::Mat OpencvGpuLaplacian_PinnedMem(const cv::Mat& input_cpu_img,
                                     int kernel_size, double scale);
void ApplyLaplacianWithGaussian(
    const Npp8u* d_input, Npp8u* d_output, int width, int height,
    int step  // normalmente uguale a width, se non ci sono padding
);