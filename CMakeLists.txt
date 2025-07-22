cmake_minimum_required(VERSION 3.18)
project(cuda_library_comparision LANGUAGES CXX CUDA)

set(CMAKE_CUDA_STANDARD 17)  # Imposta C++17 anche per CUDA
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

enable_language(CUDA)
# Set the CUDA architecture 4070 Ada Lovelace
set(CMAKE_CUDA_ARCHITECTURES 90 80 75)  # Ada Lovelace, Ampere, Turing

message(STATUS "CMAKE_SOURCE_DIR: ${CMAKE_SOURCE_DIR}")

include_directories(
    ${CMAKE_SOURCE_DIR}/inc
)

# ============================
# LOGURU setup
# ============================
# separate the loguru library from the main project
# to avoid linking issues with nvcc
add_library(loguru STATIC
    ${CMAKE_SOURCE_DIR}/third_party/loguru/loguru.cpp
)

target_include_directories(loguru PUBLIC
    ${CMAKE_SOURCE_DIR}/third_party/loguru
)

add_executable(cuda_library_comparision 
    src/main.cpp
    src/laplacian_methods.cpp
    src/utils.cpp)

find_package(CUDAToolkit)
if(CUDAToolkit_FOUND)
    message(STATUS "CUDAToolkit_VERSION: ${CUDAToolkit_VERSION}")
    message(STATUS "CUDA_PATH: $ENV{CUDA_PATH}")
    message(STATUS "CUDAToolkit_INCLUDE_DIRS: ${CUDAToolkit_INCLUDE_DIRS}")
    set(CUDA_LIB_PATH "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.8/lib/x64")
    link_directories(${CUDA_LIB_PATH})
    message(STATUS "CUDA_LIB_PATH: ${CUDA_LIB_PATH}")
else()
    message(STATUS "CUDAToolkit not found")
endif()

find_package(OpenCV REQUIRED)
message(STATUS "OpenCV version: ${OpenCV_VERSION}")
message(STATUS "OpenCV_DIR: ${OpenCV_DIR}")
message(STATUS "OpenCV_LIBS: ${OpenCV_LIBS}")

target_include_directories(cuda_library_comparision PRIVATE ${OpenCV_INCLUDE_DIRS})
# Recursively find all subdirectories in Common and add them to the include path
file(GLOB_RECURSE COMMON_HEADERS ${CMAKE_SOURCE_DIR}/Common/*.h*)
set(COMMON_INCLUDE_DIRS "")
foreach(header ${COMMON_HEADERS})
    get_filename_component(dir ${header} DIRECTORY)
    list(APPEND COMMON_INCLUDE_DIRS ${dir})
endforeach()
list(REMOVE_DUPLICATES COMMON_INCLUDE_DIRS)
target_include_directories(cuda_library_comparision PRIVATE ${COMMON_INCLUDE_DIRS})
get_target_property(NPPIAL_LIB CUDA::nppial IMPORTED_LOCATION)
message(STATUS "CUDA::nppial location: ${NPPIAL_LIB}")
target_link_libraries(cuda_library_comparision PRIVATE 
    CUDA::cudart 
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
target_link_libraries(cuda_library_comparision PRIVATE ${OpenCV_LIBS})
target_link_libraries(cuda_library_comparision PRIVATE loguru)


# Post-build step to copy the OpenCV DLLs to the executable directory
if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    set(OPENCV_DLL_PATH "${OpenCV_DIR}/bin/Debug/opencv_world4110d.dll")
else()
    set(OPENCV_DLL_PATH "${OpenCV_DIR}/bin/Release/opencv_world4110.dll")
endif()

add_custom_command(TARGET cuda_library_comparision POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
    "${OPENCV_DLL_PATH}" $<TARGET_FILE_DIR:cuda_library_comparision>
    COMMENT "Copying OpenCV DLLs to executable directory"
)

set(CMAKE_VERBOSE_MAKEFILE ON)