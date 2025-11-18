#include "filters/gpu_npp_filters.hpp"

#include <cuda_runtime.h>
#include <npp.h>

#include <opencv2/opencv.hpp>

#include "loguru.hpp"

namespace gpu::filters::npp {
/**
 * @brief Check an NPP status and exit on error.
 *
 * Logs an error message using `loguru` and calls `exit(EXIT_FAILURE)` when
 * `status` is not `NPP_SUCCESS`.
 *
 * @param status The NPP return status to check.
 * @param msg A short context message to include in the log if an error occurs.
 */
void checkNppStatus(NppStatus status, const char* msg) {
    if (status != NPP_SUCCESS) {
        LOG_F(ERROR, "NPP Error at %s: %d", msg, status);
        exit(EXIT_FAILURE);
    }
}

/**
 * @brief Apply Laplacian using NPP with device pointers.
 *
 * This helper assumes both source and destination pointers point to device
 * memory. It executes `nppiFilterLaplace_8u_C1R` using a 3x3 Laplacian mask.
 *
 * @param d_src Device pointer to source image data (byte pointer).
 * @param step Line step in bytes for the image (typically `image.step`).
 * @param roi Region of interest size (`NppiSize`) defining width/height.
 * @param d_dst Device pointer to destination buffer where the Laplacian
 *              result will be written. Must be allocated by the caller with
 *              at least `step * roi.height` bytes.
 */
void Laplacian_NPP_Device(Npp8u* d_src, int step, NppiSize roi, Npp8u* d_dst) {
    checkNppStatus(nppiFilterLaplace_8u_C1R(d_src, step, d_dst, step, roi,
                                            NPP_MASK_SIZE_3_X_3),
                   "Laplacian Filter (device)");
}

/**
 * @brief Apply Laplacian where input is host memory and output is on host.
 *
 * Copies the host `image` to a temporary device buffer, runs the NPP
 * Laplacian filter, copies the result back to `out_cpu_img` and frees the
 * temporary device buffers before returning.
 *
 * @param image Input image in host memory (`cv::Mat`). Expected to be single-
 *              channel 8-bit (`CV_8UC1`).
 * @param out_cpu_img Output image in host memory (`cv::Mat`) where the
 *                    Laplacian result will be written. Must be preallocated
 *                    with the same size/type as `image`.
 */
void Laplacian_NPP_Host(const cv::Mat& image, cv::Mat& out_cpu_img) {
    int width = image.cols;
    int height = image.rows;
    int step = static_cast<int>(image.step);
    NppiSize roi = {width, height};

    Npp8u* d_src = nullptr;
    Npp8u* d_dst = nullptr;
    cudaMalloc(&d_src, step * height);
    cudaMalloc(&d_dst, step * height);

    cudaMemcpy(d_src, image.data, step * height, cudaMemcpyHostToDevice);

    // Call device variant
    Laplacian_NPP_Device(d_src, step, roi, d_dst);

    // Copy back to host
    cudaMemcpy(out_cpu_img.data, d_dst, step * height, cudaMemcpyDeviceToHost);

    cudaFree(d_src);
    cudaFree(d_dst);
}

/**
 * @brief Apply Gaussian smoothing using NPP with device pointers.
 *
 * This helper assumes both source and destination pointers point to device
 * memory. It executes `nppiFilterGauss_8u_C1R` using a 3x3 Gaussian mask.
 *
 * @param d_src Device pointer to source image data (byte pointer).
 * @param step Line step in bytes for the image (typically `image.step`).
 * @param roi Region of interest size (`NppiSize`) defining width/height.
 * @param d_dst Device pointer to destination buffer where the smoothed image
 *              will be written. Must be allocated by the caller with at least
 *              `step * roi.height` bytes.
 */
void GaussianFilter_NPP_Device(Npp8u* d_src, int step, NppiSize roi,
                               Npp8u* d_dst) {
    checkNppStatus(nppiFilterGauss_8u_C1R(d_src, step, d_dst, step, roi,
                                          NPP_MASK_SIZE_3_X_3),
                   "Gaussian Filter (device)");
}

/**
 * @brief Apply Gaussian smoothing where input is host memory and output is on
 * device.
 *
 * Copies the host `image` to a temporary device buffer, runs the NPP Gaussian
 * filter, and leaves the smoothed image in `d_dst` (device memory). The
 * destination buffer `d_dst` must be allocated by the caller.
 *
 * @param image Input image in host memory (`cv::Mat`). Expected to be single-
 *              channel 8-bit (`CV_8UC1`).
 * @param d_dst Device pointer where the smoothed image will be written. Must
 *              have at least `image.step * image.rows` bytes allocated.
 *
 * @note This helper performs a host->device copy internally and frees the
 *       temporary device source buffer before returning.
 */
void GaussianFilter_NPP_Host(const cv::Mat& image, Npp8u* d_dst) {
    int width = image.cols;
    int height = image.rows;
    int step = static_cast<int>(image.step);
    NppiSize roi = {width, height};

    Npp8u* d_src = nullptr;
    cudaMalloc(&d_src, step * height);
    cudaMemcpy(d_src, image.data, step * height, cudaMemcpyHostToDevice);

    // Call device variant
    GaussianFilter_NPP_Device(d_src, step, roi, d_dst);

    cudaFree(d_src);
}

}  // namespace gpu::filters::npp