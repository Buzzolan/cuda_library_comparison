#include "filters/gpu_npp_filters.hpp"

#include <cuda_runtime.h>
#include <npp.h>

#include <opencv2/opencv.hpp>

#include "loguru.hpp"
#include "utils.hpp"

namespace gpu::filters::npp {
void checkNppStatus(NppStatus status, const char* msg) {
    if (status != NPP_SUCCESS) {
        LOG_F(ERROR, "NPP Error at %s: %d", msg, status);
        exit(EXIT_FAILURE);
    }
}

void ApplyLaplacianWithGaussian(const cv::Mat& image, cv::Mat& out_cpu_img) {
    int width = image.cols;
    int height = image.rows;
    int step = static_cast<int>(image.step);  // step in bytes
    NppiSize roi = {width, height};

    // Allocate device memory
    Npp8u* d_input = nullptr;
    Npp8u* d_output = nullptr;
    cudaMalloc(&d_input, step * height);
    cudaMalloc(&d_output, step * height);

    // Copy image data to device
    cudaMemcpy(d_input, image.data, step * height, cudaMemcpyHostToDevice);
    Stopwatch stopwatch;

    // Allochiamo buffer temporaneo per smoothing
    Npp8u* d_smooth;
    cudaMalloc(&d_smooth, step * height);

    // Step 1: Gaussian smoothing
    checkNppStatus(nppiFilterGauss_8u_C1R(d_input, step, d_smooth, step, roi,
                                          NPP_MASK_SIZE_3_X_3),
                   "Gaussian Filter");
    LOG_F(INFO, "Gaussian smoothing Time: %.2f ms", stopwatch.Elapsed_ms());

    // Step 2: Laplacian filtering
    checkNppStatus(nppiFilterLaplace_8u_C1R(d_smooth, step, d_output, step, roi,
                                            NPP_MASK_SIZE_3_X_3),
                   "Laplacian Filter");

    // Cleanup
    cudaFree(d_smooth);
    LOG_F(INFO, "Laplacian with Gaussian Time: %.2f ms", stopwatch.Elapsed_ms());

    // Copy result back to host
    cudaMemcpy(out_cpu_img.data, d_output, step * height, cudaMemcpyDeviceToHost);

    // Free device memory
    cudaFree(d_input);
    cudaFree(d_output);
}
}  // namespace gpu::filters::npp