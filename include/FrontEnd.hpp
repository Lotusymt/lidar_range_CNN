#pragma once

#include "slam/Types.hpp"

namespace slam {

/**
 * @brief Odometry front-end for LiDAR + IMU.
 *
 * This class consumes synchronized LiDAR scans (with attached IMU segments)
 * and produces relative pose estimates (PoseDelta) between consecutive frames.
 *
 * Internally, it can:
 *  - Integrate IMU to produce a rough motion prior.
 *  - Run scan matching / ICP between the previous and current LiDAR scans.
 *  - Maintain a running global odometry estimate.
 */
class FrontEnd {
public:
    /**
     * @brief Construct a new FrontEnd with identity initial pose.
     */
    FrontEnd();

    /**
     * @brief Process a single LiDAR scan and return relative motion from
     *        the previous scan.
     *
     * @param scan
     *        Current LiDAR scan, including its IMU segment from the previous
     *        scan to this one.
     *
     * @return PoseDelta
     *         For the first scan, the delta will be identity (zero motion).
     *         For subsequent scans, it is the estimated transform from
     *         the previous pose to the current pose plus an information matrix.
     */
    PoseDelta processFrame(const LidarScan& scan);

    /**
     * @brief Get the current global odometry estimate.
     *
     * @return Pose3D
     *         Pose of the most recent frame in the odometry/world frame.
     */
    Pose3D currentPose() const { return current_pose_; }

private:
    /// @brief Whether a previous scan has been seen.
    bool has_prev_scan_ = false;

    /// @brief Last processed LiDAR scan.
    LidarScan prev_scan_;

    /// @brief Current estimated global pose (odometry chain).
    Pose3D current_pose_;

    /**
     * @brief Integrate IMU segment to obtain a motion prior.
     *
     * The integration is typically coarse (e.g., small-angle approximation)
     * and used as an initial guess for scan matching. A more sophisticated
     * implementation can include gravity, bias estimation, and preintegration.
     *
     * @param imu_segment
     *        Vector of IMU samples between the previous and current scan.
     *
     * @return Pose3D
     *         Rough relative pose from previous frame to current frame.
     */
    Pose3D integrateImuPrior(const ImuSample& imu_prev,
        const ImuSample& imu_curr) const;

    /**
     * @brief Run scan matching / ICP between two LiDAR scans.
     *
     * Uses the previous and current LiDAR scans, along with an initial
     * pose guess (e.g., from IMU integration), to estimate the relative
     * transform. The result is packaged as a PoseDelta for the pose graph.
     *
     * @param prev
     *        Previous LiDAR scan.
     * @param curr
     *        Current LiDAR scan.
     * @param initial_guess
     *        Initial guess for the transform from prev to curr.
     *
     * @return PoseDelta
     *         Estimated transform and its information matrix.
     */
    PoseDelta icpBetweenScans(const LidarScan& prev,
                              const LidarScan& curr,
                              const Pose3D& initial_guess) const;
};

}  // namespace slam
