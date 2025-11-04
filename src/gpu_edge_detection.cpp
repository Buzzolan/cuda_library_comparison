#include "gpu_edge_detection.hpp"

#include "loguru.hpp"
#include "utils.hpp"

namespace gpu_edge_detection {
/**
 * @brief Apply a Laplacian filter to the input image using OpenCV (CPU
 * implementation).
 *
 * This function applies a Laplacian operator to enhance edges in a grayscale
 * image and stores the result in @p output_image.
 *
 * @param input_image Input grayscale image.
 * @param output_image Output image containing the Laplacian result.
 * @param kernel_size Size of the Laplacian kernel (must be odd and positive).
 * @param contrast_factor Scaling factor applied to the Laplacian result.
 */
void OpencvLaplacian(const cv::Mat& input_image, cv::Mat& laplacian_output,
                     int kernel_size, double contrast_factor) {
    Stopwatch stopwatch;

    cv::Laplacian(input_image, laplacian_output, CV_8U, kernel_size,
                  contrast_factor);

    LOG_F(INFO, "Laplacian opencv CPU Time: %.2f ms", stopwatch.Elapsed_ms());
}
}  // namespace gpu_edge_detection