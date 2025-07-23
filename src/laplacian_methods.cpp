#include <cuda_runtime.h>
#include <npp.h>

#include <iostream>
#include <opencv2/cudaarithm.hpp>
#include <opencv2/cudafilters.hpp>
#include <opencv2/cudaimgproc.hpp>

#undef LOGURU_WITH_STREAMS
#include "laplacian_methods.hpp"
#include "loguru.hpp"
#include "utils.hpp"

cv::Mat OpencvCpuLaplacian(const cv::Mat& input_image, int kernel_size, double contrast_factor) {
    cv::Mat laplacian_output;
    Stopwatch stopwatch;

    cv::Laplacian(input_image, laplacian_output, CV_8U, kernel_size, contrast_factor);
    LOG_F(INFO, "Laplacian opencv CPU Time: %.2f ms", stopwatch.Elapsed_ms());

    return laplacian_output;
}

cv::Mat OpencvGpuLaplacian(const cv::Mat& input_cpu_img, int kernel_size, double scale) {
    // Controlla se CUDA è disponibile
    if (cv::cuda::getCudaEnabledDeviceCount() == 0) {
        throw std::runtime_error("CUDA non disponibile o OpenCV non compilato con supporto CUDA.");
    }
    Stopwatch stopwatch;
    cv::TickMeter total_timer;
    total_timer.start();
    // Upload su GPU
    cv::cuda::GpuMat d_input, d_output;
    d_input.upload(input_cpu_img);

    LOG_F(INFO, "Laplacian opencv GPU Upload Time: %.2f ms", stopwatch.Elapsed_ms());
    stopwatch.Restart();

    // Crea filtro Laplaciano
    auto laplacian_filter =
        cv::cuda::createLaplacianFilter(d_input.type(),   // tipo input
                                        d_output.type(),  // tipo output
                                        kernel_size,      // dimensione kernel
                                        scale             // fattore di scala (contrast factor)
        );

    LOG_F(INFO, "Laplacian opencv GPU Filter Creation Time: %.2f ms", stopwatch.Elapsed_ms());
    stopwatch.Restart();

    // Applica filtro
    laplacian_filter->apply(d_input, d_output);

    LOG_F(INFO, "Laplacian opencv GPU Time: %.2f ms", stopwatch.Elapsed_ms());
    stopwatch.Restart();
    // Scarica risultato su CPU

    cv::Mat result;
    d_output.download(result);

    LOG_F(INFO, "Laplacian opencv GPU Download Time: %.2f ms", stopwatch.Elapsed_ms());
    total_timer.stop();
    LOG_F(INFO, "Total Laplacian opencv GPU Time: %.2f ms", total_timer.getTimeMilli());

    return result;
}

void checkNppStatus(NppStatus status, const char* msg) {
    if (status != NPP_SUCCESS) {
        std::cerr << "NPP Error at " << msg << ": " << status << std::endl;
        exit(EXIT_FAILURE);
    }
}

void ApplyLaplacianWithGaussian(const Npp8u* d_input, Npp8u* d_output, int width, int height,
                                int step  // normalmente uguale a width, se non ci sono padding
) {
    Stopwatch stopwatch;
    NppiSize roi = {width, height};

    // Allochiamo buffer temporaneo per smoothing
    Npp8u* d_smooth;
    cudaMalloc(&d_smooth, step * height);

    // Step 1: Gaussian smoothing
    checkNppStatus(nppiFilterGauss_8u_C1R(d_input, step, d_smooth, step, roi, NPP_MASK_SIZE_3_X_3),
                   "Gaussian Filter");
    LOG_F(INFO, "Gaussian smoothing Time: %.2f ms", stopwatch.Elapsed_ms());

    // Step 2: Laplacian filtering
    checkNppStatus(
        nppiFilterLaplace_8u_C1R(d_smooth, step, d_output, step, roi, NPP_MASK_SIZE_3_X_3),
        "Laplacian Filter");

    // Cleanup
    cudaFree(d_smooth);
    LOG_F(INFO, "Laplacian with Gaussian Time: %.2f ms", stopwatch.Elapsed_ms());
}

cv::Mat OpencvGpuLaplacian_PinnedMem(const cv::Mat& input_cpu_img, int kernel_size, double scale) {
    if (cv::cuda::getCudaEnabledDeviceCount() == 0) {
        throw std::runtime_error("CUDA not available or OpenCV built without CUDA support.");
    }

    cv::TickMeter total_timer;
    total_timer.start();

    // -------------------------
    // 🔹 Step 1: Upload using pinned memory
    // -------------------------
    cv::TickMeter upload_timer;
    upload_timer.start();

    // Convert input to pinned memory (page-locked)
    cv::cuda::HostMem pinned_input(input_cpu_img, cv::cuda::HostMem::PAGE_LOCKED);

    // Upload to GPU from pinned memory (faster than normal memory)
    cv::cuda::GpuMat d_input(pinned_input);

    upload_timer.stop();
    LOG_F(INFO, "Upload using pinned memory: %.2f ms", upload_timer.getTimeMilli());

    // -------------------------
    // 🔹 Step 2: Create Laplacian filter
    // -------------------------
    cv::TickMeter filter_create_timer;
    filter_create_timer.start();

    // Use d_input.type() for input and output type
    cv::cuda::GpuMat d_output;
    auto laplacian_filter =
        cv::cuda::createLaplacianFilter(d_input.type(), d_input.type(), kernel_size, scale);

    filter_create_timer.stop();
    LOG_F(INFO, "Filter creation time: %.2f ms", filter_create_timer.getTimeMilli());

    // -------------------------
    // 🔹 Step 3: Apply filter
    // -------------------------
    cv::TickMeter filter_timer;
    filter_timer.start();

    laplacian_filter->apply(d_input, d_output);

    filter_timer.stop();
    LOG_F(INFO, "Laplacian filtering time: %.2f ms", filter_timer.getTimeMilli());

    // -------------------------
    // 🔹 Step 4: Download result back to CPU
    // -------------------------
    cv::TickMeter download_timer;
    download_timer.start();

    cv::Mat result;
    d_output.download(result);  // You could also download into another HostMem

    download_timer.stop();
    LOG_F(INFO, "Download time: %.2f ms", download_timer.getTimeMilli());

    total_timer.stop();
    LOG_F(INFO, "Total GPU Laplacian time: %.2f ms", total_timer.getTimeMilli());

    return result;
}
