#include <iostream>
#include <string>
#include <opencv2/opencv.hpp>
#include <cuda_runtime.h>
#include <npp.h>
#undef LOGURU_WITH_STREAMS
#include "loguru.hpp"
#include "laplacian_methods.hpp"
#include "utils.hpp"


int main(int argc, char** argv) {
    if (argc < 3) {
        std::cerr << "Usage: " << argv[0] << " <image_path> <kernel_size> <contrast_factor>\n";
        return 1;
    }
    cv::Mat result_cpu_opencv;
    cv::Mat result_gpu_opencv;
    cv::Mat result_gpu_opencv_pinned;
    std::string image_path = argv[1];
    int kernel_size = std::stoi(argv[2]);
    double contrast_factor = std::stod(argv[3]);

    cv::Mat image = cv::imread(image_path, cv::IMREAD_GRAYSCALE);
    if (image.empty()) {
        std::cerr << "Error: Could not open image at " << image_path << "\n";
        return 1;
    }
    LOG_F(INFO, "Image size: %d x %d", image.cols, image.rows);

    result_cpu_opencv = OpencvCpuLaplacian(image, kernel_size, contrast_factor);
    cv::imwrite("output_laplacian_opencv_cpu.png", result_cpu_opencv);

    // print GPU information
    cudaDeviceProp deviceProp;
    cudaError_t  err = cudaGetDeviceProperties(&deviceProp, 0);
    if (err != cudaSuccess) {
        std::cerr << "Failed to get device properties: " << cudaGetErrorString(err) << std::endl;
        return -1;
    }

    LOG_F(INFO, "GPU name: %s", deviceProp.name);
    LOG_F(INFO, "Total global memory: %zu bytes", deviceProp.totalGlobalMem);
    LOG_F(INFO, "Shared memory per block: %zu bytes", deviceProp.sharedMemPerBlock);
    LOG_F(INFO, "Max threads per block: %d", deviceProp.maxThreadsPerBlock);

     try {
        result_gpu_opencv = OpencvGpuLaplacian(image, kernel_size, contrast_factor);
        result_gpu_opencv_pinned = OpencvGpuLaplacian_PinnedMem(image, kernel_size, contrast_factor);
        cv::imwrite("output_laplacian_opencv_gpu.png", result_gpu_opencv);
    } catch (const std::exception& ex) {
        std::cerr << "Errore: " << ex.what() << "\n";
    }

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
    int step = static_cast<int>(image.step); // step in bytes
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
