#include "filters/gpu_opencv_filters.hpp"

#include <cuda_runtime.h>

#include <opencv2/cudafilters.hpp>

#include "loguru.hpp"
#include "utils.hpp"

namespace gpu::filters::opencv {
/**
 * @brief Applies a Laplacian filter on an image using OpenCV 4.11.0 with CUDA
 * acceleration.
 *
 * This function performs a Laplacian edge-detection operation on the input image
 * by leveraging OpenCV's CUDA module. The image is first uploaded to GPU memory,
 * processed using `cv::cuda::createLaplacianFilter`, and then downloaded back to
 * CPU memory. Execution times for each processing stage (upload, filter creation,
 * filtering, and download) are logged for performance analysis.
 *
 * The function supports **only kernel sizes 1 and 3**, as limited by
 * OpenCV 4.11.0’s CUDA implementation.
 *
 * @param input_cpu_img Input image stored in CPU memory (`cv::Mat`).
 *                      Must be a valid OpenCV matrix with a type supported by
 *                      CUDA filters (e.g., `CV_8UC1`, `CV_8UC3`, etc.).
 * @param out_cpu_img Output image stored in CPU memory (`cv::Mat`).
 *                    The result of the Laplacian filtering will be stored here.
 * @param kernel_size Size of the Laplacian kernel. Must be either **1 or 3**.
 *                    Other values are not supported by the CUDA backend in
 *                    OpenCV 4.11.0.
 * @param scale Scaling factor applied to the computed Laplacian result.
 *              This value can be used to control the contrast or response
 *              strength of the filter.
 *
 * @throws std::runtime_error If CUDA is not available or OpenCV was not compiled
 *                            with CUDA support.
 */
void ApplyLaplacian(const cv::Mat& input_cpu_img, cv::Mat& out_cpu_img,
                    int kernel_size, double scale) {
    LOG_F(INFO, "------------------OpencvGpuLaplacian----------------------");
    CheckGpuSupportOrThrow();
    Stopwatch stopwatch, total_stopwatch;
    // Upload su GPU
    cv::cuda::GpuMat d_input, d_output;
    d_input.upload(input_cpu_img);

    LOG_F(INFO, "Laplacian opencv GPU Upload Time: %.2f ms",
          stopwatch.Elapsed_ms());
    stopwatch.Restart();

    // Crea filtro Laplaciano
    CV_Assert(kernel_size == 1 || kernel_size == 3);
    auto laplacian_filter = cv::cuda::createLaplacianFilter(
        d_input.type(),   // tipo input
        d_output.type(),  // tipo output
        kernel_size,      // dimensione kernel
        scale             // fattore di scala (contrast factor)
    );

    LOG_F(INFO, "Laplacian opencv GPU Filter Creation Time: %.2f ms",
          stopwatch.Elapsed_ms());
    stopwatch.Restart();

    // Apply filter
    laplacian_filter->apply(d_input, d_output);

    LOG_F(INFO, "Laplacian opencv GPU Time: %.2f ms", stopwatch.Elapsed_ms());

    stopwatch.Restart();
    // download output back to CPU
    d_output.download(out_cpu_img);
    LOG_F(INFO, "Laplacian opencv GPU Download Time: %.2f ms",
          stopwatch.Elapsed_ms());
    LOG_F(INFO, "Laplacian opencv GPU Total Time: %.2f ms",
          total_stopwatch.Elapsed_ms());
}

/**
 * @brief Applies a Laplacian filter on an image using OpenCV 4.11.0 with CUDA
 * acceleration and pinned (page-locked) host memory for optimized data transfer.
 *
 * This function performs Laplacian edge detection using OpenCV’s CUDA module,
 * similar to `OpencvGpuLaplacian`, but leverages **pinned (page-locked) memory**
 * for faster CPU–GPU data transfers. The input image is first wrapped in
 * `cv::cuda::HostMem` with the `PAGE_LOCKED` flag, which allows more efficient
 * DMA transfers between CPU and GPU memory.
 *
 * The function supports **only kernel sizes 1 and 3**, as limited by OpenCV
 * 4.11.0’s CUDA Laplacian implementation. Execution times for each major step
 * (upload, filter creation, filtering, and download) are logged for performance
 * benchmarking.
 *
 * @param input_cpu_img Input image stored in CPU memory (`cv::Mat`).
 *                      The image will be internally copied into pinned memory
 *                      before being uploaded to the GPU.
 * @param out_cpu_img Output image stored in CPU memory (`cv::Mat`).
 *                    The filtered result will be downloaded here after GPU
 *                    processing.
 * @param kernel_size Size of the Laplacian kernel. Must be either **1 or 3**.
 *                    Other values are not supported by OpenCV’s CUDA backend.
 * @param scale Scaling factor applied to the computed Laplacian result.
 *              Controls the intensity or contrast of the output edges.
 *
 * @throws std::runtime_error If CUDA is not available or OpenCV was not compiled
 *                            with CUDA support.
 */
void ApplyLaplacianWithPinnedMem(const cv::Mat& input_cpu_img,
                                 cv::Mat& out_cpu_img, int kernel_size,
                                 double scale) {
    LOG_F(INFO, "---------------OpencvGpuLaplacian_PinnedMem-------------------");
    CheckGpuSupportOrThrow();
    Stopwatch total_stopwatch, stopwatch;
    // Convert input to pinned memory (page-locked)
    cv::cuda::HostMem pinned_input(input_cpu_img, cv::cuda::HostMem::PAGE_LOCKED);

    // Upload to GPU from pinned memory (faster than normal memory)
    cv::cuda::GpuMat d_input(pinned_input);

    LOG_F(INFO, "Upload time (pinned memory): %.2f ms", stopwatch.Elapsed_ms());
    stopwatch.Restart();

    // Use d_input.type() for input and output type
    cv::cuda::GpuMat d_output;
    auto laplacian_filter = cv::cuda::createLaplacianFilter(
        d_input.type(), d_input.type(), kernel_size, scale);

    LOG_F(INFO, "Filter creation time: %.2f ms", stopwatch.Elapsed_ms());
    stopwatch.Restart();

    laplacian_filter->apply(d_input, d_output);

    LOG_F(INFO, "Laplacian filtering time: %.2f ms", stopwatch.Elapsed_ms());
    stopwatch.Restart();

    d_output.download(
        out_cpu_img);  // You could also download into another HostMem

    LOG_F(INFO, "Download time: %.2f ms", stopwatch.Elapsed_ms());
    LOG_F(INFO, "Total time (pinned memory): %.2f ms",
          total_stopwatch.Elapsed_ms());
}
}  // namespace gpu::filters::opencv