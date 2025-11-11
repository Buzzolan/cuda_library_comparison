#include "laplacian_methods.hpp"

#include <cuda_runtime.h>
#include <npp.h>

#include <iostream>
#include <opencv2/cudaarithm.hpp>
#include <opencv2/cudafilters.hpp>
#include <opencv2/cudaimgproc.hpp>

#include "loguru.hpp"
#include "utils.hpp"

void checkNppStatus(NppStatus status, const char* msg) {
    if (status != NPP_SUCCESS) {
        std::cerr << "NPP Error at " << msg << ": " << status << std::endl;
        exit(EXIT_FAILURE);
    }
}

void ApplyLaplacianWithGaussian(
    const Npp8u* d_input, Npp8u* d_output, int width, int height,
    int step  // normalmente uguale a width, se non ci sono padding
) {
    Stopwatch stopwatch;
    NppiSize roi = {width, height};

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
}
