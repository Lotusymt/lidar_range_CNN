// RangeImageProjector.cpp
#include "RangeImageProjector.hpp"

#include <cmath>
#include <limits>

namespace slam {

RangeImageProjector::RangeImageProjector(const RangeImageConfig& cfg)
    : cfg_(cfg)
{
}

void RangeImageProjector::project(const LidarScan& scan, RangeImage& img) const
{
    const int H = cfg_.height;
    const int W = cfg_.width;

    img.height = H;
    img.width  = W;
    img.timestamp = scan.timestamp;

    // Initialize with 0.0f to mean "no return"
    img.data.assign(static_cast<std::size_t>(H) * W, 0.0f);

    const double v_fov_up   = cfg_.v_fov_up;
    const double v_fov_down = cfg_.v_fov_down;
    const double v_fov_total = v_fov_up - v_fov_down;

    for (const auto& p : scan.points) {
        const double x = p.x();
        const double y = p.y();
        const double z = p.z();

        const double r = std::sqrt(x * x + y * y + z * z);
        if (r <= 0.0) {
            continue;
        }

        // Horizontal angle (yaw) in [-pi, pi]
        const double yaw = std::atan2(y, x);

        // Vertical angle (pitch)
        const double xy_norm = std::sqrt(x * x + y * y);
        const double pitch = std::atan2(z, xy_norm);

        // Map pitch to row index [0, H-1]
        //  pitch = v_fov_down -> row = H-1
        //  pitch = v_fov_up   -> row = 0
        double v = (pitch - v_fov_down) / v_fov_total;   // [0, 1] ideally
        double row_f = (1.0 - v) * (H - 1);              // flip vertical

        int row = static_cast<int>(std::round(row_f));
        if (row < 0 || row >= H) {
            continue; // outside vertical FOV
        }

        // Map yaw [-pi, pi] to col index [0, W-1]
        double h = 0.5 * (yaw / M_PI + 1.0);  // [-pi,pi] -> [0,1]
        double col_f = h * (W - 1);

        int col = static_cast<int>(std::round(col_f));
        if (col < 0 || col >= W) {
            continue; // just in case
        }

        const std::size_t idx = static_cast<std::size_t>(row) * W + col;

        float& cell = img.data[idx];
        // keep nearest point if multiple points hit the same pixel
        if (cell == 0.0f || r < static_cast<double>(cell)) {
            cell = static_cast<float>(r);
        }
    }
}

} // namespace slam
