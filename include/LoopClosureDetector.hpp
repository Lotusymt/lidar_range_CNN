#pragma once

#include "slam/Types.hpp"
#include "slam/LoopClosureNet.hpp"
#include "slam/RangeImageProjector.hpp"
#include <vector>
#include <memory>

namespace slam {

/**
 * @brief Loop closure detector using range images and neural network.
 *
 * This class manages keyframes (range images) and periodically checks
 * for loop closures by comparing the current frame with past keyframes
 * using the LoopClosureNet neural network.
 *
 * Architecture note: Loop closure detection is typically a separate
 * component from the frontend, running periodically and feeding results
 * to the backend (pose graph optimizer).
 */
class LoopClosureDetector {
public:
    /**
     * @brief Construct a new LoopClosureDetector.
     *
     * @param model_path Path to the TorchScript model file (.pt)
     * @param probability_threshold Minimum probability to accept a loop closure (default: 0.5)
     * @param keyframe_interval Add a keyframe every N frames (default: 10)
     * @param min_separation Minimum frames between current and candidate keyframe (default: 50)
     * @param max_candidates Maximum number of candidates to check per detection (default: 20)
     */
    LoopClosureDetector(const std::string& model_path,
                       float probability_threshold = 0.5f,
                       int keyframe_interval = 10,
                       int min_separation = 50,
                       int max_candidates = 20);

    /**
     * @brief Process a new frame and check for loop closures.
     *
     * This method:
     * 1. Projects the LiDAR scan to a range image
     * 2. Decides if this should be a keyframe
     * 3. If it's a keyframe, checks against past keyframes for loop closures
     *
     * @param scan Current LiDAR scan
     * @param current_frame_id Current frame index (0-based)
     * @param current_node_index Current node index in pose graph (should match frame_id if one node per frame)
     * @param current_pose Current global pose estimate
     * @param projector Range image projector to use
     *
     * @return Vector of loop closure candidates (empty if none found)
     */
    std::vector<LoopClosureCandidate> processFrame(const LidarScan& scan,
                                                   int current_frame_id,
                                                   int current_node_index,
                                                   const Pose3D& current_pose,
                                                   const RangeImageProjector& projector);
    
    /**
     * @brief Get the keyframe ID (node index in pose graph) for a given frame ID.
     *
     * @param frame_id Frame index
     * @return Keyframe index in pose graph, or -1 if not found
     */
    int getKeyframeNodeIndex(int frame_id) const;

    /**
     * @brief Get the number of stored keyframes.
     */
    size_t numKeyframes() const { return keyframes_.size(); }

    /**
     * @brief Clear all stored keyframes (useful for resetting).
     */
    void clearKeyframes() { keyframes_.clear(); }

private:
    /// @brief Neural network for loop closure detection
    std::unique_ptr<LoopClosureNet> net_;

    /// @brief Minimum probability to accept a loop closure
    float probability_threshold_;

    /// @brief Add a keyframe every N frames
    int keyframe_interval_;

    /// @brief Minimum frames between current and candidate
    int min_separation_;

    /// @brief Maximum candidates to check per detection
    int max_candidates_;

    /// @brief Stored keyframe information
    struct Keyframe {
        int frame_id;           // Original frame index
        int node_index;          // Node index in pose graph
        RangeImage range_image;
        Pose3D pose;            // Global pose at this keyframe
    };

    std::vector<Keyframe> keyframes_;
};

} // namespace slam

