// src/LoopClosureNet.cpp
#include "slam/LoopClosureNet.hpp"
#include <stdexcept>
#include <cstring>   // std::memcpy

namespace slam {

LoopClosureNet::LoopClosureNet(const std::string& model_path)
{
    // Load TorchScript module
    module_ = torch::jit::load(model_path);
    module_.eval();  // inference mode
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
