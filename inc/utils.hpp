#pragma once
#include <chrono>

double computeSSIM(const cv::Mat& img1, const cv::Mat& img2);
double getMSE(const cv::Mat& img1, const cv::Mat& img2);

class Stopwatch {
public:
    Stopwatch() { Restart(); }

    void Restart() {
        start_time_ = std::chrono::high_resolution_clock::now();
    }

    double Elapsed_ms() const {
        auto now = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double, std::milli> elapsed = now - start_time_;
        return elapsed.count();
    }

private:
    std::chrono::high_resolution_clock::time_point start_time_;
};
