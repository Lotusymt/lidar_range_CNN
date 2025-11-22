// RangeImageProjector.hpp
#pragma once

#include "Types.hpp"

namespace slam {

/**
 * @brief Configuration for projecting LiDAR points into a range image.
 *
 * Vertical FOV is split between v_fov_up and v_fov_down (in radians).
 * Example for a 64x1024 image and ~26° vertical FOV:
 *   height = 64;
 *   width  = 1024;
 *   v_fov_up   = 2.0 * M_PI / 180.0;   // +2°
 *   v_fov_down = -24.9 * M_PI / 180.0; // -24.9°
 */
struct RangeImageConfig {
    int height = 64;    ///< Number of rows
    int width  = 1024;  ///< Number of columns

    double v_fov_up   = 2.0 * M_PI / 180.0;
    double v_fov_down = -24.9 * M_PI / 180.0;
};

class RangeImageProjector {
public:
    explicit RangeImageProjector(const RangeImageConfig& cfg);

    /**
     * @brief Project a LiDAR scan into a range image.
     *
     * @param[in]  scan  Input LiDAR scan (points in sensor frame).
     * @param[out] img   Output range image (range in meters).
     *
     * Pixels with no valid point will have value 0.0f.
     * If multiple points map to the same pixel, we keep the closest.
     */
    void project(const LidarScan& scan, RangeImage& img) const;

private:
    RangeImageConfig cfg_;
};

} // namespace slam
