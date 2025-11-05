#include <cuda_runtime.h>
#include <npp.h>

#include <opencv2/opencv.hpp>

#include "ReadSettings.hpp"
#include "cpu_edge_detection.hpp"
#include "gpu_edge_detection.hpp"
#include "laplacian_methods.hpp"
#include "loguru.hpp"
#include "utils.hpp"

int main(int argc, char* argv[]) {
    // set loguru for debugging
    loguru::init(argc, argv);
    loguru::add_file("log.txt", loguru::Append, loguru::Verbosity_INFO);

    // read settings
    auto [image, kernel_size, contrast_factor] =
        settings::InitSettings("settings.json");

    // -----------------------------------------------------------------------------------
    // CPU Implementation
    // -----------------------------------------------------------------------------------

    // OpenCV CPU Laplacian
    cv::Mat result_cpu_opencv;
    cpu_edge_detection::OpencvLaplacian(image, result_cpu_opencv, kernel_size,
                                        contrast_factor);
    cv::imwrite("output_laplacian_opencv_cpu.png", result_cpu_opencv);

    // -----------------------------------------------------------------------------------
    // GPU Implementations
    // -----------------------------------------------------------------------------------

    // OpenCV GPU Laplacian
    cv::Mat result_gpu_opencv;
    gpu_edge_detection::OpencvGpuLaplacian(image, result_gpu_opencv, kernel_size,
                                           contrast_factor);

    // OpenCV GPU Laplacian with Pinned Memory
    // cv::Mat result_gpu_opencv_pinned;
    // OpencvGpuLaplacian_PinnedMem(image, result_gpu_opencv_pinned, kernel_size,
    //                              contrast_factor);

    // -----------------------------------------------------------------------------------
    // Results Saving and Comparison
    // -----------------------------------------------------------------------------------

    cv::imwrite("output_laplacian_opencv_gpu.png", result_gpu_opencv);

    // Compute and print MSE and SSIM
    double mse = getMSE(result_cpu_opencv, result_gpu_opencv);
    double ssim = computeSSIM(result_cpu_opencv, result_gpu_opencv);

    LOG_F(INFO, "MSE == 0 means no difference between images.");
    LOG_F(INFO, "SSIM == 1 means images are identical.");
    LOG_F(INFO, "MSE: %.4f", mse);
    LOG_F(INFO, "SSIM: %.4f", ssim);

    // NPP gaussina + laplacian

    int width = image.cols;
    int height = image.rows;
    int step = static_cast<int>(image.step);  // step in bytes
    NppiSize roi = {width, height};

    cv::Mat img_out(height, width, CV_8UC1);

    // Allocate device memory
    Npp8u* d_src = nullptr;
    Npp8u* d_dst = nullptr;
    cudaMalloc(&d_src, step * height);
    cudaMalloc(&d_dst, step * height);

    // Copy image data to device
    cudaMemcpy(d_src, image.data, step * height, cudaMemcpyHostToDevice);

    // Apply laplacian with Gaussian smoothing
    ApplyLaplacianWithGaussian(d_src, d_dst, width, height, step);

    // Copy result back to host
    cudaMemcpy(img_out.data, d_dst, step * height, cudaMemcpyDeviceToHost);

    // Free device memory
    cudaFree(d_src);
    cudaFree(d_dst);

    // Save the result
    cv::imwrite("output_laplacian_npp.png", img_out);

    // compute and print MSE and SSIM for NPP result
    double mse_npp = getMSE(result_cpu_opencv, img_out);
    double ssim_npp = computeSSIM(result_cpu_opencv, img_out);

    LOG_F(INFO, "NPP MSE cpu_opencv vs NPP: %.4f", mse_npp);
    LOG_F(INFO, "NPP SSIM cpu_opencv vs NPP: %.4f", ssim_npp);

    return 0;
}