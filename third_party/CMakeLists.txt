cmake_minimum_required(VERSION 3.15)

# ============================
# LOGURU setup
# ============================
add_library(loguru STATIC  ${CMAKE_CURRENT_LIST_DIR}/loguru/loguru.cpp)
target_include_directories(loguru PUBLIC ${CMAKE_CURRENT_LIST_DIR}/loguru)
