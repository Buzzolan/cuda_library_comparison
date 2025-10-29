#include "ReadSettings.hpp"

#include <cuda_runtime.h>

#include <opencv2/opencv.hpp>

#include "loguru.hpp"

namespace settings {
std::tuple<cv::Mat, uint8_t, double> InitSettings(
    const std::filesystem::path& filename) {
    ReadSettings reader(filename);

    std::string image_path = reader.GetValue<std::string>("Settings.image_path",
                                                          "data/grey-sloth.png");
    uint8_t kernel_size = reader.GetValue<int>("Settings.kernel_size", 3);
    double contrast_factor =
        reader.GetValue<double>("Settings.contrast_factor", 1.0);

    LOG_F(INFO, "----- Settings loaded: -----");
    LOG_F(INFO, "Image Path: %s", image_path.c_str());
    LOG_F(INFO, "Kernel Size: %d", kernel_size);
    LOG_F(INFO, "Contrast Factor: %.2f", contrast_factor);

    cv::Mat image = cv::imread(image_path, cv::IMREAD_GRAYSCALE);
    if (image.empty()) {
        std::cerr << "Error: Could not open image at " << image_path << "\n";
        throw std::runtime_error("Could not open image at " + image_path);
    }
    LOG_F(INFO, "Image size: %d x %d", image.cols, image.rows);

    PrintGPUInfo();
    return std::make_tuple(image, kernel_size, contrast_factor);
}

void PrintGPUInfo() {
    // print GPU information
    cudaDeviceProp deviceProp;
    cudaError_t err = cudaGetDeviceProperties(&deviceProp, 0);
    if (err != cudaSuccess) {
        throw std::runtime_error("Failed to get device properties" +
                                 std::string(cudaGetErrorString(err)));
    }

    LOG_F(INFO, "----- GPU Information -----");
    LOG_F(INFO, "GPU name: %s", deviceProp.name);
    LOG_F(INFO, "Total global memory: %zu bytes", deviceProp.totalGlobalMem);
    LOG_F(INFO, "Shared memory per block: %zu bytes",
          deviceProp.sharedMemPerBlock);
    LOG_F(INFO, "Max threads per block: %d", deviceProp.maxThreadsPerBlock);
    LOG_F(INFO, "---------------------------");
}
}  // namespace settings