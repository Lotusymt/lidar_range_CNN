#pragma once

#include <vector>
#include <Eigen/Core>
#include <Eigen/Geometry>

namespace slam {

/**
 * @brief 6-DOF pose in 3D (SE(3)).
 *
 * Represents a rigid-body transform with translation and rotation.
 * Convention: transforms a point from local frame into parent/world frame.
 */
struct Pose3D {
    /// @brief Translation component in meters (x, y, z).
    Eigen::Vector3d t;

    /// @brief Orientation as a unit quaternion (w, x, y, z).
    Eigen::Quaterniond q;
};

/**
 * @brief Relative motion (edge) between two consecutive poses.
 *
 * This is the quantity produced by the odometry front-end and inserted
 * as an edge into the pose-graph / optimizer.
 */
struct PoseDelta {
    /// @brief Relative transform from previous pose to current pose.
    Pose3D T_prev_to_curr;

    /**
     * @brief Information matrix (inverse covariance) of this measurement.
     *
     * 6x6 matrix in (rot, trans) space (e.g., [rx, ry, rz, tx, ty, tz]).
     * Higher values mean higher confidence.
     */
    Eigen::Matrix<double, 6, 6> info;
};

/**
 * @brief One IMU measurement sample.
 *
 * Contains timestamp, linear acceleration, and angular velocity in the
 * IMU / body frame.
 */
struct ImuSample {
    /// @brief Timestamp in seconds.
    double timestamp;

    /// @brief Linear acceleration in body frame [m/s^2].
    Eigen::Vector3d accel;

    /// @brief Angular velocity in body frame [rad/s].
    Eigen::Vector3d gyro;
};

/**
 * @brief One LiDAR scan plus attached IMU segment.
 *
 * The IMU segment should cover the time interval between the previous
 * LiDAR frame and this one, so that the front-end can integrate it.
 */
struct LidarScan {
    /// @brief 3D points in the LiDAR sensor frame.
    std::vector<Eigen::Vector3d> points;

    /// @brief Timestamp of this scan in seconds.
    double timestamp;

    /**
     * @brief Synchronized IMU/OXTS sample at this LiDAR timestamp.
     *
     * This is the 10 Hz OXTS packet from oxts/data/XXXXXX.txt in the
     * *_sync dataset (not the original 100 Hz stream).
     */
     ImuSample imu;
};

/**
 * @brief 2D LiDAR range image representation.
 *
 * Stores a dense H x W image in row-major order (r * width + c),
 * where each pixel contains a scalar (e.g., range).
 */
struct RangeImage {
    /// @brief Flattened range image data in row-major order.
    std::vector<float> data;

    /// @brief Image height (number of rows).
    int height = 0;

    /// @brief Image width (number of columns).
    int width = 0;

    /// @brief Timestamp associated with this range image in seconds.
    double timestamp = 0.0;
};

/**
 * @brief Loop closure candidate result.
 *
 * Represents a potential loop closure detection between two keyframes.
 */
struct LoopClosureCandidate {
    /// @brief Keyframe ID of the past frame that matches the current frame.
    int keyframe_id;

    /// @brief Probability of loop closure (from neural network).
    float probability;

    /// @brief Estimated relative transform from past keyframe to current keyframe.
    /// This matches PoseDelta convention: T_prev_to_curr (from past to current).
    /// In a more sophisticated system, you might run ICP here to refine it.
    Pose3D T_past_to_curr;
};

}  // namespace slam
