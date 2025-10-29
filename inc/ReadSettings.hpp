#pragma once

#include <filesystem>
#include <fstream>
#include <json.hpp>
#include <opencv2/opencv.hpp>
#include <sstream>
#include <stdexcept>
#include <string>
#include <tuple>

namespace settings {
void PrintGPUInfo();
std::tuple<cv::Mat, uint8_t, double> InitSettings(
    const std::filesystem::path& filename);

class ReadSettings {
   public:
    explicit ReadSettings(const std::filesystem::path& filename) {
        std::ifstream file(filename);
        if (!file.is_open()) {
            throw std::runtime_error("Could not open settings file: " +
                                     filename.string());
        }
        try {
            file >> settings_json_;  // Parse JSON content
        } catch (const nlohmann::json::parse_error& e) {
            throw std::runtime_error("JSON parse error in file " +
                                     filename.string() + ": " + e.what());
        }
    }

    // Methods:

    const nlohmann::json& GetSettings() const noexcept { return settings_json_; }

    template <typename T>
    T GetValue(const std::string& key) const {
        const nlohmann::json* current = &settings_json_;
        std::istringstream ss(key);
        std::string token;

        while (std::getline(ss, token, '.')) {
            if (!current->contains(token)) {
                throw std::runtime_error("Missing key: " + token);
            }
            current = &((*current)[token]);
        }

        return current->get<T>();
    }

    template <typename T>
    T GetValue(const std::string& path, const T& default_value) const noexcept {
        try {
            return GetValue<T>(path);
        } catch (...) {
            return default_value;
        }
    }

   private:
    nlohmann::json settings_json_;
};
}  // namespace settings