// KittiLoader.hpp
#pragma once

#include "Types.hpp"

#include <string>
#include <vector>

namespace slam {

class KittiLoader {
public:
    /**
     * @brief Construct loader from explicit LiDAR and IMU directories.
     *
     * Example:
     *   velodyne_dir = "D:\\kitti_lidar\\00\\velodyne"
     *   imu_dir      = "D:\\lidar_imu\\01\\imu"
     */
    KittiLoader(const std::string& velodyne_dir,
                const std::string& imu_dir);

    /**
     * @brief Load next LiDAR scan + IMU sample.
     *
     * @param[out] scan  Filled with points, timestamp, and imu.
     * @return false when there are no more matched frames.
     */
    bool loadNextScan(LidarScan& scan);

    /// @brief Reset internal index so that the next call starts from frame 0.
    void reset();

    /// @brief Number of matched frames (min(#lidar, #imu)).
    std::size_t size() const { return size_; }

private:
    std::vector<std::string> lidar_files_;  // sorted .bin paths
    std::vector<std::string> imu_files_;    // sorted .txt paths
    std::size_t index_ = 0;
    std::size_t size_  = 0;                 // matched frames

    static bool loadLidarBin(const std::string& path,
                             std::vector<Eigen::Vector3d>& points);

    static bool loadImuTxt(const std::string& path,
                           ImuSample& imu);
};

} // namespace slam
