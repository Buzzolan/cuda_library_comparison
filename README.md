# cuda_library_comparision Project

A C++ project for high-performance image preprocessing using **OpenCV with CUDA** for Laplacian edge detection.  
Includes **CPU fallback**, large image support, and **image comparison using MSE and SSIM** (self-implemented).

---

## Features

- Laplacian edge detection with **CUDA acceleration** or **CPU fallback**
- Handles very large images (e.g., 12000 × 33000 pixels)
- Timing and logging via [loguru](https://github.com/emilk/loguru)
- Image comparison tools:
  - **MSE (Mean Squared Error)**
  - **SSIM (Structural Similarity Index)** — custom implementation
- Clean switch between **Debug** and **Release** modes

---

## Build Instructions

### 📦 Requirements

- CMake ≥ 3.18
- OpenCV (with CUDA modules built)
- CUDA Toolkit (≥ 11.x)
- C++17 compiler

### ⚙️ CMake Build

```bash
# Create and enter build directory
mkdir build && cd build

# Configure (choose Debug or Release)
cmake -DCMAKE_BUILD_TYPE=Release ..

# Build the project
cmake --build .

---

### Run

```bash
./Laplacian_cuda_project input_image.png laplacian_kernel_size contrast_factor
```

Example:

```bash
./Laplacian_cuda_project image.png 3 1.0
```

---

## What Is the Laplacian Filter?

The **Laplacian filter** is a mathematical operation used in image processing to **highlight edges and fine details**.

### Simple Explanation

* It looks at **how fast pixel values change** in all directions (up/down/left/right).
* If the change is big (i.e. edges or textures), it **amplifies** that area.
* If the area is smooth, it stays dark or unchanged.

### What It Does

| Original     | After Laplacian           |
| ------------ | ------------------------- |
| Smooth image | Only edges remain visible |

### Why Use It?

* Helps detect cracks, lines, or shapes in images
* Good for analyzing small details in high-res images
* Essential in **preprocessing** before computer vision tasks

---

### ⚙️ Parameters

* **Kernel Size**
  Controls how large of a neighborhood to consider for edge detection.

  * GPU: only `1` or `3` allowed
  * CPU: can use larger values like `5` or `7` for smoother results

* **Contrast Factor**
  Boosts or reduces the strength of edges.
  Small values (e.g. `0.03`) help make subtle edges visible in large, smooth images.

---

## Image Quality Metrics

This project supports evaluating output image similarity using:

* **MSE** (Mean Squared Error)
* **SSIM** (Structural Similarity Index, self-implemented)

These metrics are used to compare GPU and CPU Laplacian results.

---

## Project Structure

```
.
├── CMakeLists.txt
├── src/
│   ├── main.cpp
│   ├── laplacian_methods.cpp
│   └── utils.cpp
├── inc/
│   └── laplacian_methods.hpp
└── third_party/
    └── loguru/
```

---

## Notes

* CUDA Laplacian filter supports only **kernel sizes of 1 or 3**.
* Larger kernels (e.g., 5 or 7) are supported via CPU.
* No inversion is applied in any preprocessing step.
* Debug and Release modes automatically configure DLL paths in post-build steps.

---

## License

This project is intended for internal development, educational, or benchmarking use only.
