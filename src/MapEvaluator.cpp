#include "MapEvaluator.hpp"
#include <algorithm>
#include <cmath>
#include <iostream>
#include <iomanip>
#include <limits>

#ifdef PCL_FOUND
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/kdtree/kdtree_flann.h>
#endif

namespace slam {

MapEvaluator::MapEvaluator(double inlier_threshold, double max_search_radius)
    : inlier_threshold_(inlier_threshold)
    , max_search_radius_(max_search_radius)
#ifdef PCL_FOUND
    , map_cloud_(new pcl::PointCloud<pcl::PointXYZ>())
    , kdtree_built_(false)
#endif
{
}

MapEvaluator::Metrics MapEvaluator::evaluate(
    const MapManager& map_manager,
    const std::map<int, LidarScan>& keyframe_scans,
    const std::vector<Pose3D>& optimized_poses,
    const std::vector<std::pair<int, int>>& keyframe_indices) const
{
    Metrics metrics;
    metrics.inlier_threshold = inlier_threshold_;
    
    const std::vector<Eigen::Vector3d>& map_points = map_manager.getMapPoints();
    
    if (map_points.empty()) {
        std::cerr << "MapEvaluator: Map is empty, cannot evaluate\n";
        return metrics;
    }

#ifdef PCL_FOUND
    // Build KD-tree once for fast nearest neighbor search (O(N log N) build, O(log N) per query)
    // This replaces O(N) linear search with O(log N) tree search = 100-1000x faster for large maps
    if (!kdtree_built_) {
        std::cout << "[MapEvaluator] Building KD-tree from " << map_points.size() << " map points...\n";
        std::cout.flush();
        buildKDTree(map_points);
        std::cout << "[MapEvaluator] KD-tree built. Using fast nearest neighbor search (O(log N) per query).\n";
        std::cout.flush();
    }
#else
    std::cout << "[MapEvaluator] WARNING: PCL not available, using slow linear search (O(N) per query).\n";
    std::cout << "[MapEvaluator] Consider building with PCL for 100-1000x speedup on large maps.\n";
    std::cout.flush();
#endif

    std::vector<double> all_distances;
    size_t total_inliers = 0;
    size_t total_points = 0;

    // For large maps, sample points to speed up evaluation
    // Evaluate every Nth point instead of all points
    const int SAMPLE_RATE = (map_points.size() > 500000) ? 10 : 1;  // Sample every 10th point for large maps
    if (SAMPLE_RATE > 1) {
        std::cout << "[MapEvaluator] Sampling every " << SAMPLE_RATE << " points for evaluation.\n";
        std::cout.flush();
    }

    size_t keyframe_count = 0;
    // Evaluate each keyframe
    for (const auto& [frame_id, node_idx] : keyframe_indices) {
        // Find corresponding scan
        auto scan_it = keyframe_scans.find(frame_id);
        if (scan_it == keyframe_scans.end()) {
            continue;  // Skip if scan not found
        }

        // Check if pose exists
        if (node_idx < 0 || node_idx >= static_cast<int>(optimized_poses.size())) {
            continue;
        }

        const LidarScan& scan = scan_it->second;
        const Pose3D& pose = optimized_poses[node_idx];

        // Progress output
        keyframe_count++;
        if (keyframe_count % 10 == 0) {
            std::cout << "[MapEvaluator] Evaluating keyframe " << keyframe_count 
                      << "/" << keyframe_indices.size() << "...\n";
            std::cout.flush();
        }

        // Transform scan points to world coordinates
        // Sample points if map is large
        size_t point_idx = 0;
        for (const auto& point : scan.points) {
            if (point_idx % SAMPLE_RATE == 0) {  // Sample every Nth point
                Eigen::Vector3d world_point = pose.q * point + pose.t;
                
                // Find nearest map point
                double distance = findNearestDistance(world_point, map_points);
                
                all_distances.push_back(distance);
                total_points++;
                
                if (distance <= inlier_threshold_) {
                    total_inliers++;
                }
            }
            point_idx++;
        }
        
        metrics.num_keyframes++;
    }

    metrics.total_points = total_points;

    if (all_distances.empty()) {
        return metrics;
    }

    // Compute average distance
    double sum = 0.0;
    for (double d : all_distances) {
        sum += d;
    }
    metrics.avg_point_to_map_distance = sum / all_distances.size();

    // Compute median distance
    std::vector<double> sorted_distances = all_distances;
    std::sort(sorted_distances.begin(), sorted_distances.end());
    size_t mid = sorted_distances.size() / 2;
    if (sorted_distances.size() % 2 == 0) {
        metrics.median_point_to_map_distance = 
            (sorted_distances[mid - 1] + sorted_distances[mid]) / 2.0;
    } else {
        metrics.median_point_to_map_distance = sorted_distances[mid];
    }

    // Compute inlier ratio
    metrics.inlier_ratio = (total_points > 0) ? 
        (static_cast<double>(total_inliers) / total_points) : 0.0;

    return metrics;
}

#ifdef PCL_FOUND
void MapEvaluator::buildKDTree(const std::vector<Eigen::Vector3d>& map_points) const
{
    map_cloud_->clear();
    map_cloud_->points.resize(map_points.size());
    
    for (size_t i = 0; i < map_points.size(); ++i) {
        map_cloud_->points[i].x = static_cast<float>(map_points[i].x());
        map_cloud_->points[i].y = static_cast<float>(map_points[i].y());
        map_cloud_->points[i].z = static_cast<float>(map_points[i].z());
    }
    
    map_cloud_->width = static_cast<uint32_t>(map_cloud_->points.size());
    map_cloud_->height = 1;
    map_cloud_->is_dense = true;
    
    map_kdtree_.setInputCloud(map_cloud_);
    kdtree_built_ = true;
}
#endif

double MapEvaluator::findNearestDistance(
    const Eigen::Vector3d& query_point,
    const std::vector<Eigen::Vector3d>& map_points) const
{
#ifdef PCL_FOUND
    // Use KD-tree if available (much faster: O(log N) vs O(N))
    if (kdtree_built_ && map_cloud_ && !map_cloud_->empty()) {
        pcl::PointXYZ search_pt;
        search_pt.x = static_cast<float>(query_point.x());
        search_pt.y = static_cast<float>(query_point.y());
        search_pt.z = static_cast<float>(query_point.z());
        
        std::vector<int> indices(1);
        std::vector<float> sq_dists(1);
        
        int found = map_kdtree_.nearestKSearch(search_pt, 1, indices, sq_dists);
        
        if (found > 0) {
            double distance = std::sqrt(static_cast<double>(sq_dists[0]));
            return std::min(distance, max_search_radius_);
        } else {
            return max_search_radius_;
        }
    }
#endif

    // Fallback: linear search (slow but works without PCL)
    if (map_points.empty()) {
        return max_search_radius_;
    }

    double min_distance = max_search_radius_;
    double min_distance_sq = max_search_radius_ * max_search_radius_;  // Use squared distance for efficiency
    
    // Simple linear search (for small maps or when PCL not available)
    // OPTIMIZATION: Use squared distance to avoid sqrt() in inner loop
    for (const auto& map_point : map_points) {
        Eigen::Vector3d diff = query_point - map_point;
        double dist_sq = diff.squaredNorm();  // Faster than .norm() (no sqrt)
        
        if (dist_sq < min_distance_sq) {
            min_distance_sq = dist_sq;
            min_distance = std::sqrt(dist_sq);
        }
        
        // Early exit if we found a very close point
        if (min_distance_sq < 0.0001) {  // 0.01^2 = 0.0001
            break;
        }
    }

    // Cap at max_search_radius
    return std::min(min_distance, max_search_radius_);
}

void MapEvaluator::printMetrics(const Metrics& metrics)
{
    std::cout << "\n";
    std::cout << "========================================\n";
    std::cout << "      Map Quality Evaluation\n";
    std::cout << "========================================\n";
    std::cout << "Keyframes evaluated: " << metrics.num_keyframes << "\n";
    std::cout << "Total points evaluated: " << metrics.total_points << "\n";
    std::cout << "\n";
    std::cout << "Point-to-Map Distance:\n";
    std::cout << "  Average: " << std::fixed << std::setprecision(4) 
              << metrics.avg_point_to_map_distance << " m\n";
    std::cout << "  Median:  " << std::fixed << std::setprecision(4) 
              << metrics.median_point_to_map_distance << " m\n";
    std::cout << "\n";
    std::cout << "Inlier Statistics:\n";
    std::cout << "  Threshold: " << std::fixed << std::setprecision(3) 
              << metrics.inlier_threshold << " m\n";
    std::cout << "  Inlier Ratio: " << std::fixed << std::setprecision(2) 
              << (metrics.inlier_ratio * 100.0) << "%\n";
    std::cout << "========================================\n";
    std::cout << "\n";
    
    // Interpretation
    if (metrics.avg_point_to_map_distance < 0.1) {
        std::cout << "[OK] Excellent map quality (avg distance < 0.1m)\n";
    } else if (metrics.avg_point_to_map_distance < 0.2) {
        std::cout << "[OK] Good map quality (avg distance < 0.2m)\n";
    } else if (metrics.avg_point_to_map_distance < 0.5) {
        std::cout << "[WARN] Moderate map quality (avg distance < 0.5m)\n";
    } else {
        std::cout << "[FAIL] Poor map quality (avg distance >= 0.5m)\n";
    }
    
    if (metrics.inlier_ratio > 0.8) {
        std::cout << "[OK] High consistency (>80% inliers)\n";
    } else if (metrics.inlier_ratio > 0.6) {
        std::cout << "[WARN] Moderate consistency (60-80% inliers)\n";
    } else {
        std::cout << "[FAIL] Low consistency (<60% inliers)\n";
    }
    std::cout << "\n";
}

} // namespace slam

