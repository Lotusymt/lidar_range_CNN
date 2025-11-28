#pragma once
#include "Types.hpp"
#include <vector>
#include <string>

namespace slam {

/**
 * @brief Map manager that builds a point cloud map from keyframes.
 * 
 * Uses voxel downsampling to save memory. Only integrates keyframes,
 * not every frame, to reduce computational cost.
 */
class MapManager {
public:
    /**
     * @brief Construct a new MapManager.
     * 
     * @param voxel_size Voxel grid leaf size in meters (e.g., 0.2 = 20cm)
     *                    Larger values = more downsampling = less memory
     */
    explicit MapManager(double voxel_size = 0.2);

    /**
     * @brief Integrate a keyframe scan into the map.
     * 
     * Transforms points from sensor frame to world frame using the pose,
     * then adds them to the map with voxel downsampling.
     * 
     * @param pose Optimized pose from pose graph (world frame)
     * @param scan LiDAR scan in sensor frame
     */
    void integrateKeyframe(const Pose3D& pose, const LidarScan& scan);

    /**
     * @brief Get the current map as a point cloud.
     * 
     * @return Vector of 3D points in world frame (already downsampled)
     */
    const std::vector<Eigen::Vector3d>& getMapPoints() const { return map_points_; }

    /**
     * @brief Get number of points in the map.
     */
    size_t numPoints() const { return map_points_.size(); }

    /**
     * @brief Save map to PLY file for visualization.
     * 
     * @param filename Output filename (e.g., "map.ply")
     */
    void saveToPLY(const std::string& filename) const;

    /**
     * @brief Clear the map.
     */
    void clear() { map_points_.clear(); }

private:
    std::vector<Eigen::Vector3d> map_points_;  ///< Global point cloud (world frame)
    double voxel_size_;                         ///< Voxel grid resolution

    /**
     * @brief Apply voxel downsampling to a point cloud.
     * 
     * Groups points into voxels and keeps one point per voxel (centroid).
     * 
     * @param points Input points
     * @return Downsampled points
     */
    std::vector<Eigen::Vector3d> voxelDownsample(const std::vector<Eigen::Vector3d>& points) const;
};

} // namespace slam
