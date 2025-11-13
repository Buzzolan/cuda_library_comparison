#include <cuda_runtime.h>
#include <npp.h>

#include <opencv2/opencv.hpp>

#include "ReadSettings.hpp"
#include "filters/cpu_opencv_filters.hpp"
#include "filters/gpu_npp_filters.hpp"
#include "filters/gpu_opencv_filters.hpp"
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
    cpu::filters::opencv::ApplyLaplacian(image, result_cpu_opencv, kernel_size,
                                         contrast_factor);
    cv::imwrite("output_laplacian_opencv_cpu.png", result_cpu_opencv);

    // -----------------------------------------------------------------------------------
    // GPU Implementations
    // -----------------------------------------------------------------------------------

    // OpenCV GPU Laplacian
    cv::Mat result_gpu_opencv;
    gpu::filters::opencv::ApplyLaplacian(image, result_gpu_opencv, kernel_size,
                                         contrast_factor);
    cv::imwrite("output_laplacian_opencv_gpu.png", result_gpu_opencv);

    // OpenCV GPU Laplacian with Pinned Memory
    cv::Mat result_gpu_opencv_pinned;
    gpu::filters::opencv::ApplyLaplacianWithPinnedMem(
        image, result_gpu_opencv_pinned, kernel_size, contrast_factor);
    cv::imwrite("output_laplacian_opencv_gpu_pinned.png",
                result_gpu_opencv_pinned);

    // Npp GPU Laplacian with Gaussian Smoothing
    cv::Mat img_out(image.rows, image.cols, CV_8UC1);
    gpu::filters::npp::ApplyLaplacianWithGaussian(image, img_out);
    cv::imwrite("output_laplacian_npp.png", img_out);

    // -----------------------------------------------------------------------------------
    // Results Comparison
    // -----------------------------------------------------------------------------------

    // Compute and print MSE and SSIM
    double mse = getMSE(result_cpu_opencv, result_gpu_opencv);
    double ssim = computeSSIM(result_cpu_opencv, result_gpu_opencv);

    LOG_F(INFO, "MSE == 0 means no difference between images.");
    LOG_F(INFO, "SSIM == 1 means images are identical.");
    LOG_F(INFO, "MSE: %.4f", mse);
    LOG_F(INFO, "SSIM: %.4f", ssim);

    // compute and print MSE and SSIM for NPP result
    double mse_npp = getMSE(result_cpu_opencv, img_out);
    double ssim_npp = computeSSIM(result_cpu_opencv, img_out);

    LOG_F(INFO, "NPP MSE cpu_opencv vs NPP: %.4f", mse_npp);
    LOG_F(INFO, "NPP SSIM cpu_opencv vs NPP: %.4f", ssim_npp);

    return 0;
}