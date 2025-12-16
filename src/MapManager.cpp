#include "MapManager.hpp"
#include <fstream>
#include <unordered_map>
#include <algorithm>
#include <cmath>

namespace slam {

MapManager::MapManager(double voxel_size)
    : voxel_size_(voxel_size)
{
    if (voxel_size <= 0.0) {
        throw std::invalid_argument("MapManager: voxel_size must be positive");
    }
}

void MapManager::integrateKeyframe(const Pose3D& pose, const LidarScan& scan)
{
    // Transform points from sensor frame to world frame
    std::vector<Eigen::Vector3d> world_points;
    world_points.reserve(scan.points.size());

    for (const auto& point : scan.points) {
        // Transform: P_world = R * P_sensor + t
        Eigen::Vector3d world_point = pose.q * point + pose.t;
        world_points.push_back(world_point);
    }

    // Add to map
    map_points_.insert(map_points_.end(), world_points.begin(), world_points.end());

    // Periodically downsample to save memory
    // Downsample every 10 keyframes or when map gets large
    if (map_points_.size() > 100000) {
        map_points_ = voxelDownsample(map_points_);
    }
}

std::vector<Eigen::Vector3d> MapManager::voxelDownsample(
    const std::vector<Eigen::Vector3d>& points) const
{
    if (points.empty()) {
        return points;
    }

    // Use hash map to group points by voxel
    // Voxel key: (floor(x/voxel_size), floor(y/voxel_size), floor(z/voxel_size))
    struct VoxelKey {
        int x, y, z;
        bool operator==(const VoxelKey& other) const {
            return x == other.x && y == other.y && z == other.z;
        }
    };

    struct VoxelKeyHash {
        std::size_t operator()(const VoxelKey& k) const {
            // Simple hash function
            return std::hash<int>()(k.x) ^ (std::hash<int>()(k.y) << 1) ^ (std::hash<int>()(k.z) << 2);
        }
    };

    std::unordered_map<VoxelKey, std::vector<Eigen::Vector3d>, VoxelKeyHash> voxel_map;

    // Group points by voxel
    for (const auto& point : points) {
        VoxelKey key;
        key.x = static_cast<int>(std::floor(point.x() / voxel_size_));
        key.y = static_cast<int>(std::floor(point.y() / voxel_size_));
        key.z = static_cast<int>(std::floor(point.z() / voxel_size_));

        voxel_map[key].push_back(point);
    }

    // Compute centroid for each voxel
    std::vector<Eigen::Vector3d> downsampled;
    downsampled.reserve(voxel_map.size());

    for (const auto& [key, voxel_points] : voxel_map) {
        if (voxel_points.empty()) continue;

        // Compute centroid
        Eigen::Vector3d centroid = Eigen::Vector3d::Zero();
        for (const auto& pt : voxel_points) {
            centroid += pt;
        }
        centroid /= static_cast<double>(voxel_points.size());

        downsampled.push_back(centroid);
    }

    return downsampled;
}

void MapManager::saveToPLY(const std::string& filename) const
{
    std::ofstream file(filename);
    if (!file.is_open()) {
        throw std::runtime_error("MapManager: Cannot open file for writing: " + filename);
    }

    // Write PLY header
    file << "ply\n";
    file << "format ascii 1.0\n";
    file << "element vertex " << map_points_.size() << "\n";
    file << "property float x\n";
    file << "property float y\n";
    file << "property float z\n";
    file << "end_header\n";

    // Write points
    for (const auto& point : map_points_) {
        file << point.x() << " " << point.y() << " " << point.z() << "\n";
    }

    file.close();
}

} // namespace slam

