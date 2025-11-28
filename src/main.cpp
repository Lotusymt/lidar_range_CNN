#include "Types.hpp"
#include "KittiLoader.hpp"
#include "RangeImageProjector.hpp"
#include "FrontEnd.hpp"
#include "PoseGraph.hpp"
#include "LoopClosureDetector.hpp"
#include "MapManager.hpp"
#include "MapEvaluator.hpp"
#include "PoseEvaluator.hpp"
#include "ErrorHandler.hpp"

#include <iostream>
#include <string>
#include <cstdlib>  // for std::getenv
#include <map>      // for storing keyframe scans
#include <fstream>  // for checking ground truth file
#include <stdexcept> // for std::exception
#include <vector>   // for paths_to_try
#ifdef _WIN32
#include <direct.h>  // for _getcwd on Windows
#include <filesystem> // for path conversion
#define getcwd _getcwd
#else
#include <unistd.h>  // for getcwd on Linux/Mac
#endif

// Windows-specific: Force early initialization of standard streams
// This helps ensure output works even if DLLs fail to load
#ifdef _WIN32
#include <windows.h>
#include <io.h>
#include <fcntl.h>
static int init_stdio() {
    // Force initialization of stdout/stderr
    std::cout << std::flush;
    std::cerr << std::flush;
    return 0;
}
static int _dummy = init_stdio();
#endif

int main(int argc, char* argv[]) {
    // Add debug output at the very start
    // Use both cout and cerr, and flush immediately
    std::cout << "[DEBUG] Program starting...\n" << std::flush;
    std::cerr << "[DEBUG] Program starting (stderr)...\n" << std::flush;
    
    try {
    // Parse command line arguments for data paths
    // Priority: 1) Command-line args, 2) Environment variables, 3) Defaults
    std::string velodyne_dir = "D:\\odom_dataset\\kitti_lidar\\00\\velodyne";
    std::string imu_dir = "D:\\odom_dataset\\lidar_imu\\00\\imu";
    
    // Check environment variables first (if command-line args not provided)
    if (argc < 3) {
        const char* env_velodyne = std::getenv("KITTI_VELODYNE_DIR");
        const char* env_imu = std::getenv("KITTI_IMU_DIR");
        if (env_velodyne) velodyne_dir = env_velodyne;
        if (env_imu) imu_dir = env_imu;
    }
    
    // Command-line arguments override everything
    if (argc >= 3) {
        velodyne_dir = argv[1];
        imu_dir = argv[2];
    } else if (argc == 2) {
        std::cerr << "Usage: " << argv[0] << " <velodyne_dir> <imu_dir>\n";
        std::cerr << "Or set environment variables: KITTI_VELODYNE_DIR, KITTI_IMU_DIR\n";
        std::cerr << "Using default paths (may not exist)\n";
    }
    
    std::cout << "LiDAR directory: " << velodyne_dir << "\n";
    std::cout << "IMU directory: " << imu_dir << "\n";
    std::cout.flush();
    
    // 1) Data loader
    std::cout << "[DEBUG] Creating KittiLoader...\n";
    std::cout.flush();
    slam::KittiLoader loader(velodyne_dir, imu_dir);
    std::cout << "[DEBUG] KittiLoader created successfully.\n";
    std::cout.flush();

    // 2) Range image projector config
    slam::RangeImageConfig cfg;
    cfg.height = 64;
    cfg.width  = 1024;
    // cfg.v_fov_up / cfg.v_fov_down = defaults are okay for now

    slam::RangeImageProjector projector(cfg);

    // 3) Front-end (odometry)
    slam::FrontEnd frontend;

    // 4) Loop closure detector
    // Try multiple possible model file locations
    std::string model_path;
    std::vector<std::string> paths_to_try = {
        "model.pt",                                             // Simple name in executable directory (PREFERRED - shortest path)
        "pair_range_cnn_kitti_00_10.pt",                       // In executable directory
        "src/pair_range_cnn_kitti_00_10.pt",                   // From project root
        "build/bin/pair_range_cnn_kitti_00_10.pt",            // CMake copy location
        "build/bin/Debug/pair_range_cnn_kitti_00_10.pt",       // Debug build location
        "build/bin/Release/pair_range_cnn_kitti_00_10.pt",     // Release build location
        "../../src/pair_range_cnn_kitti_00_10.pt",            // From build/bin/Debug
        "../../../src/pair_range_cnn_kitti_00_10.pt",          // From build/bin/Debug/Release
    };
    
    bool found = false;
    for (const auto& path : paths_to_try) {
        std::ifstream model_check(path, std::ios::binary);
        if (model_check.good()) {
            // Convert to absolute path with proper Windows formatting
            #ifdef _WIN32
            try {
                std::filesystem::path fs_path(path);
                if (fs_path.is_relative()) {
                    fs_path = std::filesystem::absolute(fs_path);
                }
                model_path = fs_path.string();  // Converts to native path format (backslashes on Windows)
            } catch (...) {
                model_path = path;  // Fallback to original path
            }
            #else
            model_path = path;
            #endif
            model_check.close();
            found = true;
            break;
        }
        model_check.close();
    }
    
    if (!found) {
        std::cerr << "Error: Model file not found. Tried:\n";
        for (size_t i = 0; i < paths_to_try.size(); ++i) {
            std::cerr << "  " << (i+1) << ". " << paths_to_try[i] << "\n";
        }
        std::cerr << "\nCurrent working directory: ";
        char cwd[1024];
        if (getcwd(cwd, sizeof(cwd)) != nullptr) {
            std::cerr << cwd << "\n";
        } else {
            std::cerr << "(unknown)\n";
        }
        return 1;
    }
    std::cout << "Using model file: " << model_path << "\n";
    std::cout << "[DEBUG] About to load loop closure model...\n";
    std::cout.flush();
    
    slam::LoopClosureDetector loop_detector(
        model_path,
        0.5f,    // probability threshold
        10,      // keyframe_interval: minimum frames between keyframes (constraint)
        50,      // min separation (at least 50 frames apart)
        20,      // max candidates to check
        true,    // use_motion_based: use motion-based keyframe selection
        1.0,     // min_translation: create keyframe when moved > 1.0 meters
        0.2      // min_rotation: create keyframe when rotated > 0.2 radians (~11.5 degrees)
    );
    std::cout << "[DEBUG] LoopClosureDetector created successfully.\n";
    std::cout.flush();

    // 5) Pose-graph back-end
    slam::PoseGraph graph;

    // 6) Map manager (only uses keyframes, with voxel downsampling)
    slam::MapManager map_manager(0.2);  // 20cm voxel size

    // 7) Store keyframe scans for map building (only keyframes, not every frame)
    std::map<int, slam::LidarScan> keyframe_scans;  // frame_id -> scan
    size_t prev_num_keyframes = 0;  // Track number of keyframes to detect new ones

    // Add first node (prior on initial pose)
    slam::Pose3D initial_pose;
    initial_pose.t = Eigen::Vector3d::Zero();
    initial_pose.q = Eigen::Quaterniond::Identity();
    int last_node_idx = graph.addNode(initial_pose);

    // 8) Main loop
    slam::LidarScan scan;
    int frame_count = 0;
    bool loop_closure_detected = false;

    while (loader.loadNextScan(scan)) {
        // Front-end: estimate relative motion (odometry)
        slam::PoseDelta odometry_delta;
        slam::Pose3D current_pose;
        try {
            odometry_delta = frontend.processFrame(scan);
            current_pose = frontend.currentPose();
        } catch (const std::exception& e) {
            slam::printErrorWithContext(e, "Processing frame " + std::to_string(frame_count) + " in FrontEnd");
            return 1;
        }

        // Add new node to pose graph
        int curr_node_idx = graph.addNode(current_pose);

        // Add odometry edge from last to current
        graph.addOdometryEdge(
            last_node_idx,
            curr_node_idx,
            odometry_delta
        );

        // Loop closure detection (runs periodically on keyframes)
        // Motion-based keyframe selection uses current_pose automatically
        std::vector<slam::LoopClosureCandidate> loop_candidates;
        try {
            loop_candidates = loop_detector.processFrame(scan, frame_count, curr_node_idx, current_pose, projector);
        } catch (const std::exception& e) {
            slam::printErrorWithContext(e, "Processing frame " + std::to_string(frame_count) + " in LoopClosureDetector");
            return 1;
        }

        // Store keyframe scans for map building
        // Check if this frame became a keyframe by checking if number of keyframes increased
        size_t current_num_keyframes = loop_detector.numKeyframes();
        if (current_num_keyframes > prev_num_keyframes) {
            // New keyframe was added by LoopClosureDetector
            // Use frame_count (0-indexed) to match the frame_id used in processFrame
            keyframe_scans[frame_count] = scan;  // Store scan for this keyframe
            prev_num_keyframes = current_num_keyframes;
            std::cout << "[DEBUG] Stored keyframe scan for frame " << frame_count << "\n";
        }

        // Process loop closure candidates
        for (const auto& candidate : loop_candidates) {
            // candidate.keyframe_id is already the node index in the pose graph
            int past_node_idx = candidate.keyframe_id;
            
            if (past_node_idx >= 0 && past_node_idx < static_cast<int>(graph.poses().size())) {
                // Create a PoseDelta for the loop closure edge
                slam::PoseDelta loop_edge;
                loop_edge.T_prev_to_curr = candidate.T_past_to_curr;
                
                // Higher information (confidence) for loop closures
                // Scale based on probability
                double info_scale = static_cast<double>(candidate.probability) * 10.0;
                loop_edge.info = Eigen::Matrix<double, 6, 6>::Identity() * info_scale;

                // Add loop closure edge (from past to current)
                graph.addLoopClosureEdge(past_node_idx, curr_node_idx, loop_edge);
                
                loop_closure_detected = true;
                std::cout << "[Loop Closure] Detected between frame " << past_node_idx 
                          << " and " << frame_count 
                          << " (probability: " << candidate.probability << ")\n";
            }
        }

        // Periodically run optimization if loop closures were detected
        if (loop_closure_detected && frame_count % 100 == 0) {
            std::cout << "[PoseGraph] Running optimization after loop closure detection...\n";
            graph.optimize(20);
            loop_closure_detected = false;  // Reset flag
        }

        last_node_idx = curr_node_idx;
        ++frame_count;

        if (frame_count % 10 == 0) {
            std::cout << "Processed frame " << frame_count
                      << ", current pose t = "
                      << current_pose.t.transpose() 
                      << ", keyframes: " << loop_detector.numKeyframes() << "\n";
        }
    }

    // Final optimization
    std::cout << "\n[PoseGraph] Running final optimization...\n";
    graph.optimize(50);

    // Build map from keyframes using optimized poses
    std::cout << "\n[MapManager] Building map from keyframes...\n";
    auto keyframe_indices = loop_detector.getKeyframeIndices();
    
    std::cout << "[MapManager] Found " << keyframe_indices.size() << " keyframes, " 
              << keyframe_scans.size() << " stored scans\n";
    
    int integrated_count = 0;
    for (const auto& [frame_id, node_idx] : keyframe_indices) {
        if (node_idx >= 0 && node_idx < static_cast<int>(graph.poses().size())) {
            const auto& optimized_pose = graph.poses()[node_idx];
            
            // Find corresponding scan
            auto it = keyframe_scans.find(frame_id);
            if (it != keyframe_scans.end()) {
                if (it->second.points.empty()) {
                    std::cout << "[WARN] Keyframe " << frame_id << " has empty scan (points: " 
                              << it->second.points.size() << ")\n";
                } else {
                    map_manager.integrateKeyframe(optimized_pose, it->second);
                    integrated_count++;
                }
            } else {
                std::cout << "[WARN] No scan found for keyframe frame_id=" << frame_id 
                          << ", node_idx=" << node_idx << "\n";
            }
        }
    }
    std::cout << "[MapManager] Integrated " << integrated_count << " keyframes into map\n";

    // Final downsampling
    std::cout << "[MapManager] Final voxel downsampling...\n";
    // MapManager automatically downsamples, but we can force it by clearing and rebuilding
    // For now, the periodic downsampling should be sufficient

    // Save map
    std::cout << "[MapManager] Saving map to map.ply...\n";
    map_manager.saveToPLY("map.ply");
    std::cout << "[MapManager] Map saved. Total points: " << map_manager.numPoints() << "\n";

    // Evaluate map quality (scan-to-map alignment)
    // NOTE: This can be slow for large maps. Set SKIP_MAP_EVAL=1 to skip.
    const char* skip_eval = std::getenv("SKIP_MAP_EVAL");
    if (skip_eval && std::string(skip_eval) == "1") {
        std::cout << "\n[MapEvaluator] Skipping map evaluation (SKIP_MAP_EVAL=1)\n";
    } else {
        std::cout << "\n[MapEvaluator] Evaluating map quality...\n";
        std::cout << "[MapEvaluator] This may take a while for large maps...\n";
        std::cout << "[MapEvaluator] To skip, set: $env:SKIP_MAP_EVAL=1\n";
        std::cout.flush();
        
        slam::MapEvaluator map_evaluator(0.2, 1.0);  // 0.2m inlier threshold, 1.0m max search radius
        // Reuse keyframe_indices from above (already declared on line 171)
        auto map_metrics = map_evaluator.evaluate(
            map_manager,
            keyframe_scans,
            graph.poses(),
            keyframe_indices
        );
        slam::MapEvaluator::printMetrics(map_metrics);
    }

    // Evaluate pose accuracy (ATE/RPE) if ground truth is available
    // Auto-detect ground truth file path based on velodyne_dir
    // Expected structure: D:/odom_dataset/kitti_lidar/00/velodyne -> D:/odom_dataset/gt_poses/00
    std::string ground_truth_file = "";
    
    // Extract sequence number from velodyne_dir path (e.g., "00" from ".../kitti_lidar/00/velodyne")
    size_t seq_start = velodyne_dir.find_last_of("\\/");
    if (seq_start != std::string::npos) {
        std::string parent_dir = velodyne_dir.substr(0, seq_start);
        size_t seq_end = parent_dir.find_last_of("\\/");
        if (seq_end != std::string::npos) {
            std::string seq = parent_dir.substr(seq_end + 1);
            // Construct ground truth path: D:/odom_dataset/gt_poses/00
            size_t dataset_pos = velodyne_dir.find("odom_dataset");
            if (dataset_pos != std::string::npos) {
                std::string base_path = velodyne_dir.substr(0, dataset_pos + 12);  // "D:/odom_dataset"
                ground_truth_file = base_path + "\\gt_poses\\" + seq + ".txt";
            }
        }
    }
    
    // Fallback if auto-detection fails
    if (ground_truth_file.empty()) {
        ground_truth_file = "D:\\odom_dataset\\gt_poses\\00.txt";
    }
    
    std::cout << "[PoseEvaluator] Ground truth file: " << ground_truth_file << "\n";
    
    // Check if ground truth file exists (optional - skip if not available)
    std::ifstream gt_check(ground_truth_file);
    if (gt_check.good()) {
        gt_check.close();
        std::cout << "\n[PoseEvaluator] Evaluating pose accuracy...\n";
        slam::PoseEvaluator::evaluate(graph.poses(), ground_truth_file);
    } else {
        std::cout << "\n[PoseEvaluator] Ground truth file not found. Skipping pose evaluation.\n";
        std::cout << "  To evaluate poses, set ground_truth_file in main.cpp\n";
    }

    std::cout << "\nDone. Total frames: " << frame_count 
              << ", Total nodes: " << graph.poses().size()
              << ", Total edges: " << graph.edges().size()
              << ", Keyframes: " << keyframe_indices.size()
              << ", Map points: " << map_manager.numPoints() << "\n";
    return 0;
    } catch (const std::exception& e) {
        slam::printErrorWithContext(e, "Main execution");
        return 1;
    } catch (...) {
        std::cerr << "\n═══════════════════════════════════════════════════════════\n";
        std::cerr << "FATAL ERROR: Unknown exception occurred\n";
        std::cerr << "═══════════════════════════════════════════════════════════\n";
        return 1;
    }
}
