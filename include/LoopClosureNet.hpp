#pragma once

#include "slam/Types.hpp"
#include <torch/script.h>
#include <string>

namespace slam {

/**
 * @brief C++ wrapper around the trained PairRangeCNN loop-closure model.
 *
 * This class loads a TorchScript model exported from PyTorch and provides
 * a function to evaluate the probability that two range images correspond
 * to the same place (i.e., loop-closure likelihood).
 */
class LoopClosureNet {
public:
    /**
     * @brief Construct a new LoopClosureNet and load a TorchScript model.
     *
     * @param model_path
     *        Path to the TorchScript `.pt` file created from PairRangeCNN.
     *
     * @throws std::runtime_error
     *         If the model file cannot be loaded.
     */
    explicit LoopClosureNet(const std::string& model_path);

    /**
     * @brief Compute loop-closure probability for a pair of range images.
     *
     * The input images must have the same dimensions (height, width).
     * Internally, the two images are stacked into a single tensor of shape
     * [1, 2, H, W] and passed through the network.
     *
     * @param img1
     *        First range image.
     * @param img2
     *        Second range image.
     *
     * @return float
     *         Probability in [0, 1] that the two images represent the same place.
     */
    float loopClosureProbability(const RangeImage& img1,
                                 const RangeImage& img2);

private:
    /// @brief TorchScript module representing the PairRangeCNN.
    torch::jit::script::Module module_;
};

}  // namespace slam
