#pragma once

#include "slam/Types.hpp"

namespace slam {

/**
 * @brief Projects 3D LiDAR point clouds into 2D range images.
 *
 * Given a LidarScan, this class computes a dense H x W range image using
 * spherical projection (yaw/pitch to image coordinates). The exact projection
 * parameters (FOV, resolution) must match those used during NN training.
 */
class RangeImageGenerator {
public:
    /**
     * @brief Construct a new RangeImageGenerator.
     *
     * @param height
     *        Number of rows in the output range image.
     * @param width
     *        Number of columns in the output range image.
     * @param fov_up_deg
     *        Upper vertical field-of-view limit in degrees (positive).
     * @param fov_down_deg
     *        Lower vertical field-of-view limit in degrees (negative).
     */
    RangeImageGenerator(int height,
                        int width,
                        float fov_up_deg,
                        float fov_down_deg);

    /**
     * @brief Generate a range image from a LiDAR scan.
     *
     * The algorithm typically:
     *  - Computes range and yaw/pitch for each 3D point.
     *  - Converts yaw/pitch into image coordinates (row, col).
     *  - Stores the range in the corresponding pixel, following a policy
     *    such as "keep the closest point per pixel".
     *
     * @param scan
     *        Input LiDAR scan in the sensor frame.
     *
     * @return RangeImage
     *         Output H x W range image with timestamp copied from scan.
     */
    RangeImage generate(const LidarScan& scan) const;

private:
    /// @brief Image height (rows).
    int height_;

    /// @brief Image width (columns).
    int width_;

    /// @brief Upper vertical FOV in radians.
    float fov_up_rad_;

    /// @brief Lower vertical FOV in radians.
    float fov_down_rad_;
};

}  // namespace slam
