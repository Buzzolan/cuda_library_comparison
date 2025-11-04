cmake_minimum_required(VERSION 3.18)
project(cuda_library_comparision LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(CMAKE_CUDA_STANDARD 17)  # Imposta C++17 anche per CUDA
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

#enable_language(CUDA)
set(CMAKE_CUDA_ARCHITECTURES 90 80 75)  # 4070 Ada Lovelace, Ampere, Turing

# ============================
# Include directories
# ============================
include_directories(
    ${CMAKE_SOURCE_DIR}/inc
)

add_executable(${PROJECT_NAME} 
    src/main.cpp
    src/laplacian_methods.cpp
    src/utils.cpp
    src/ReadSettings.cpp
    src/gpu_edge_detection.cpp
)

# ============================
# Find CUDA Toolkit
# ============================
find_package(CUDAToolkit)
if(CUDAToolkit_FOUND)
    message(STATUS "CUDAToolkit_VERSION: ${CUDAToolkit_VERSION}")
    message(STATUS "CUDA_PATH: $ENV{CUDA_PATH}")
    message(STATUS "CUDAToolkit_INCLUDE_DIRS: ${CUDAToolkit_INCLUDE_DIRS}")
else()
    message(STATUS "CUDAToolkit not found")
endif()

# ============================
# Find includes OpenCV
# ============================
find_package(OpenCV REQUIRED)
if(OpenCV_FOUND)
    message(STATUS "OpenCV found: ${OpenCV_VERSION}")
    message(STATUS "OpenCV version: ${OpenCV_VERSION}")
    message(STATUS "OpenCV_DIR: ${OpenCV_DIR}")
else()
    message(FATAL_ERROR "OpenCV not found")
endif()

# ==============================
# Add third-party libraries
# ==============================
add_subdirectory(third_party)

# ============================
# Link libraries
# ============================
set(CUDA_NPP_LIBS
    CUDA::nppc 
    CUDA::nppial 
    CUDA::nppif 
    CUDA::nppist 
    CUDA::nppim 
    CUDA::nppidei 
    CUDA::nppig 
    CUDA::nppicc
    CUDA::nppisu
)
target_link_libraries(${PROJECT_NAME} PRIVATE CUDA::cudart  ${CUDA_NPP_LIBS})
target_link_libraries(${PROJECT_NAME} PRIVATE ${OpenCV_LIBS})
target_link_libraries(${PROJECT_NAME} PRIVATE loguru)
target_link_libraries(${PROJECT_NAME} PRIVATE nlohmann)

# ============================
# Additional settings
# ============================
# Post-build step to copy the OpenCV DLLs to the executable directory
set(OPENCV_DLL_NAME "opencv_world4110")
if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    set(OPENCV_DLL_NAME "${OPENCV_DLL_NAME}d.dll")
else()
    set(OPENCV_DLL_NAME "${OPENCV_DLL_NAME}.dll")
endif()

set(OPENCV_DLL_PATH "${OpenCV_DIR}/bin/${CMAKE_BUILD_TYPE}/${OPENCV_DLL_NAME}")

add_custom_command(TARGET ${PROJECT_NAME} POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
    "${OPENCV_DLL_PATH}" $<TARGET_FILE_DIR:${PROJECT_NAME}>
    COMMENT "Copying OpenCV DLLs to executable directory"
)

# Copy compile_commands.json to the source directory (project root)
# This is useful for tools like clang-tidy or IDEs that need it
if(CMAKE_EXPORT_COMPILE_COMMANDS)
    execute_process(COMMAND ${CMAKE_COMMAND} -E copy_if_different
        ${CMAKE_BINARY_DIR}/compile_commands.json
        ${CMAKE_SOURCE_DIR}/compile_commands.json)
endif()

# Copy Settings.json to the build directory
add_custom_command(TARGET ${PROJECT_NAME} POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
    "${CMAKE_SOURCE_DIR}/Settings.json" $<TARGET_FILE_DIR:${PROJECT_NAME}>
    COMMENT "Copying Settings.json to executable directory"
)

