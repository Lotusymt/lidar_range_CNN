#pragma once
#include "Types.hpp"
#include <unordered_map>

namespace slam {

class MapManager {
public:
    MapManager(double resolution, int size_x, int size_y);

    void integrateScan(const Pose3D& pose, const LidarScan& scan);

    // Returns some structure you can later feed to a planner
    // For now, you can keep it internal or just a debug visualization
    // (e.g. export to PGM, or write to a point cloud file)
private:
    // internal grid representation...
};

} // namespace slam
