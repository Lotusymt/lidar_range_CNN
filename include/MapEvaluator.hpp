#pragma once

#include "Types.hpp"
#include "MapManager.hpp"
#include <vector>
#include <string>
#include <map>
#include <memory>

// PCL for KD-tree (much faster than linear search)
#ifdef PCL_FOUND
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/kdtree/kdtree_flann.h>
#endif

namespace slam {

/**
 * @brief Evaluates map quality by comparing scans with the accumulated map.
 * 
 * Since KITTI doesn't provide ground truth maps, we evaluate map consistency:
 * - Transform each keyframe scan to world coordinates using optimized poses
 * - Compare scan points against the accumulated map
 * - Compute metrics like point-to-map distance, inlier ratio, etc.
 * 
 * This measures how well the optimized poses align scans with the map.
 */
class MapEvaluator {
public:
    /**
     * @brief Evaluation metrics for map quality.
     */
    struct Metrics {
        /// @brief Average distance from scan points to nearest map points (meters)
        double avg_point_to_map_distance = 0.0;
        
        /// @brief Median distance (more robust to outliers)
        double median_point_to_map_distance = 0.0;
        
        /// @brief Percentage of scan points within threshold distance of map (inlier ratio)
        double inlier_ratio = 0.0;
        
        /// @brief Inlier threshold distance (meters)
        double inlier_threshold = 0.2;
        
        /// @brief Number of keyframes evaluated
        size_t num_keyframes = 0;
        
        /// @brief Total number of points evaluated
        size_t total_points = 0;
    };

    /**
     * @brief Construct a new MapEvaluator.
     * 
     * @param inlier_threshold Distance threshold (meters) for considering a point an inlier
     * @param max_search_radius Maximum search radius for nearest neighbor (meters)
     */
    explicit MapEvaluator(double inlier_threshold = 0.2, double max_search_radius = 1.0);
    
    /**
     * @brief Destructor (needed for PCL smart pointers)
     */
    ~MapEvaluator() = default;

    /**
     * @brief Evaluate map quality by comparing keyframe scans with the map.
     * 
     * For each keyframe:
     * 1. Transform scan points to world coordinates using optimized pose
     * 2. Find nearest map points for each scan point
     * 3. Compute distance statistics
     * 
     * @param map_manager Map manager containing the accumulated map
     * @param keyframe_scans Map of frame_id -> LidarScan for keyframes
     * @param optimized_poses Optimized poses from pose graph (indexed by node index)
     * @param keyframe_indices Vector of (frame_id, node_index) pairs for keyframes
     * 
     * @return Evaluation metrics
     */
    Metrics evaluate(const MapManager& map_manager,
                     const std::map<int, LidarScan>& keyframe_scans,
                     const std::vector<Pose3D>& optimized_poses,
                     const std::vector<std::pair<int, int>>& keyframe_indices) const;

    /**
     * @brief Print evaluation results to console.
     */
    static void printMetrics(const Metrics& metrics);

private:
    double inlier_threshold_;
    double max_search_radius_;

#ifdef PCL_FOUND
    // KD-tree for fast nearest neighbor search
    mutable pcl::PointCloud<pcl::PointXYZ>::Ptr map_cloud_;
    mutable pcl::KdTreeFLANN<pcl::PointXYZ> map_kdtree_;
    mutable bool kdtree_built_;
    
    /**
     * @brief Build KD-tree from map points (called once, cached).
     * 
     * @param map_points Map points to build tree from
     */
    void buildKDTree(const std::vector<Eigen::Vector3d>& map_points) const;
#endif

    /**
     * @brief Find nearest point in map for a given query point.
     * 
     * Uses KD-tree if PCL is available (fast), otherwise falls back to linear search.
     * 
     * @param query_point Query point in world coordinates
     * @param map_points Map points to search (only used if KD-tree not available)
     * @return Distance to nearest point, or max_search_radius_ if none found
     */
    double findNearestDistance(const Eigen::Vector3d& query_point,
                               const std::vector<Eigen::Vector3d>& map_points) const;
};

} // namespace slam

