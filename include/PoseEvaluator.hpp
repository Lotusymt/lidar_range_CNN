#pragma once

#include "Types.hpp"
#include <vector>
#include <string>

namespace slam {

/**
 * @brief Evaluates SLAM pose accuracy using ATE (Absolute Trajectory Error) and RPE (Relative Pose Error).
 * 
 * This is the standard evaluation metric for SLAM systems.
 * Does NOT evaluate map quality - only pose accuracy.
 */
class PoseEvaluator {
public:
    /**
     * @brief Load ground truth poses from KITTI format.
     * 
     * KITTI format: Each line has 12 values (3x4 transformation matrix, row-major)
     * 
     * @param filename Path to ground truth pose file
     * @return Vector of ground truth poses
     */
    static std::vector<Pose3D> loadGroundTruth(const std::string& filename);

    /**
     * @brief Compute Absolute Trajectory Error (ATE).
     * 
     * ATE measures the absolute difference between estimated and ground truth poses.
     * Computes RMSE of translation errors after aligning trajectories.
     * 
     * @param estimated Estimated poses from SLAM
     * @param ground_truth Ground truth poses
     * @return ATE in meters (RMSE)
     */
    static double computeATE(const std::vector<Pose3D>& estimated,
                             const std::vector<Pose3D>& ground_truth);

    /**
     * @brief Compute Relative Pose Error (RPE).
     * 
     * RPE measures the error in relative motion between consecutive poses.
     * More robust to global coordinate frame misalignment.
     * 
     * @param estimated Estimated poses from SLAM
     * @param ground_truth Ground truth poses
     * @param delta Number of frames between poses to compare (default: 1)
     * @return RPE in meters (RMSE of relative translation errors)
     */
    static double computeRPE(const std::vector<Pose3D>& estimated,
                             const std::vector<Pose3D>& ground_truth,
                             int delta = 1);

    /**
     * @brief Evaluate poses and print statistics.
     * 
     * @param estimated Estimated poses
     * @param ground_truth_file Path to ground truth pose file
     */
    static void evaluate(const std::vector<Pose3D>& estimated,
                        const std::string& ground_truth_file);
};

} // namespace slam

