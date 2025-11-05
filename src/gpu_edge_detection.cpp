#include "gpu_edge_detection.hpp"

#include <cuda_runtime.h>

#include <opencv2/cudafilters.hpp>

#include "loguru.hpp"
#include "utils.hpp"

namespace gpu_edge_detection {
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
 *
 * @note
 * - Tested with **OpenCV 4.11.0** compiled with **CUDA support**.
 * - Only kernel sizes of **1 and 3** are supported.
 * - Logs detailed timing information using `LOG_F` and a custom `Stopwatch`
 * utility.
 * - Upload and download operations between CPU and GPU are included in the timing
 * logs.
 */
void OpencvGpuLaplacian(const cv::Mat& input_cpu_img, cv::Mat& out_cpu_img,
                        int kernel_size, double scale) {
    // Controlla se CUDA è disponibile
    if (cv::cuda::getCudaEnabledDeviceCount() == 0) {
        throw std::runtime_error(
            "CUDA non disponibile o OpenCV non compilato con supporto CUDA.");
    }
    Stopwatch stopwatch;
    cv::TickMeter total_timer;
    total_timer.start();
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

    total_timer.stop();
    LOG_F(INFO, "Total Laplacian opencv GPU Time: %.2f ms",
          total_timer.getTimeMilli());
}
}  // namespace gpu_edge_detection