// src/LoopClosureNet.cpp
#include "LoopClosureNet.hpp"
#include <stdexcept>
#include <cstring>   // std::memcpy, strlen
#include <vector>   // for path_buffer
#include <filesystem>
#include <torch/script.h>
#include <c10/util/Exception.h>  // for c10::Error

namespace slam {

LoopClosureNet::LoopClosureNet(const std::string& model_path)
{
    // Load TorchScript module
    // Verify file exists first
    if (!std::filesystem::exists(model_path)) {
        // Try to resolve as absolute path
        std::filesystem::path fs_path(model_path);
        if (fs_path.is_relative()) {
            fs_path = std::filesystem::absolute(fs_path);
        }
        fs_path = fs_path.lexically_normal();
        
        if (!std::filesystem::exists(fs_path)) {
            throw std::runtime_error("Model file does not exist: " + model_path);
        }
    }
    
    // Try using the path directly first (as received)
    std::cout << "[LoopClosureNet] Attempting to load model..." << std::endl;
    std::cout << "[LoopClosureNet] Path: " << model_path << std::endl;
    std::cout << "[LoopClosureNet] Path length: " << model_path.length() << std::endl;
    std::cout.flush();
    
    // Verify file can be opened with standard I/O first
    {
        std::ifstream test_file(model_path, std::ios::binary);
        if (!test_file.good()) {
            throw std::runtime_error("Cannot open model file for reading: " + model_path);
        }
        test_file.close();
        std::cout << "[LoopClosureNet] File verified readable with standard I/O" << std::endl;
    }
    
    // Load TorchScript module
    // WORKAROUND for Windows: torch::jit::load() has path encoding issues
    // Create a clean, null-terminated C-string to ensure proper encoding
    
    // Convert to absolute path and normalize
    std::filesystem::path fs_path(model_path);
    if (fs_path.is_relative()) {
        fs_path = std::filesystem::absolute(fs_path);
    }
    fs_path = fs_path.lexically_normal();
    
    // Get native path string (backslashes on Windows)
    std::string native_path = fs_path.string();
    
    // Ensure string is properly null-terminated and create a fresh copy
    std::vector<char> path_buffer(native_path.begin(), native_path.end());
    path_buffer.push_back('\0');  // Explicit null terminator
    
    // Create string from buffer to ensure clean encoding
    std::string clean_path(path_buffer.data());
    
    std::cout << "[LoopClosureNet] Loading with clean path: " << clean_path << std::endl;
    std::cout << "[LoopClosureNet] Path size: " << clean_path.size() << ", c_str size: " << strlen(clean_path.c_str()) << std::endl;
    std::cout.flush();
    
    // Try loading with the clean path
    module_ = torch::jit::load(clean_path);
    module_.eval();  // inference mode
    
    std::cout << "[LoopClosureNet] Model loaded successfully!" << std::endl;
    std::cout.flush();
}

// Helper: check shapes match
static void checkRangeImageShape(const RangeImage& img1,
                                 const RangeImage& img2)
{
    if (img1.height != img2.height || img1.width != img2.width) {
        throw std::runtime_error("LoopClosureNet: range images must have same H,W");
    }
}

float LoopClosureNet::loopClosureProbability(const RangeImage& img1,
                                             const RangeImage& img2)
{
    checkRangeImageShape(img1, img2);

    const int H = img1.height;
    const int W = img1.width;

    if (static_cast<int>(img1.data.size()) != H * W ||
        static_cast<int>(img2.data.size()) != H * W) {
        throw std::runtime_error("LoopClosureNet: RangeImage.data size mismatch");
    }

    // TorchScript model expects tensor of shape [1, 2, H, W]
    torch::Tensor input = torch::zeros({1, 2, H, W}, torch::kFloat32);

    // Layout: [B, C, H, W] is contiguous in memory, so channel 0 (img1) then channel 1 (img2)
    float* ptr = input.data_ptr<float>();

    // Copy img1 into channel 0
    std::memcpy(ptr, img1.data.data(), H * W * sizeof(float));

    // Copy img2 into channel 1
    std::memcpy(ptr + H * W, img2.data.data(), H * W * sizeof(float));

    // Wrap tensor in IValue vector
    std::vector<torch::jit::IValue> inputs;
    inputs.push_back(input);

    // Forward pass
    torch::Tensor output = module_.forward(inputs).toTensor();
    // Model returns shape [1, 1] with Sigmoid, so probability in [0,1]
    output = output.squeeze();  // -> scalar tensor

    float prob = output.item<float>();
    // Clamp slightly in case of numerical noise
    if (prob < 0.0f) prob = 0.0f;
    if (prob > 1.0f) prob = 1.0f;

    return prob;
}

} // namespace slam
