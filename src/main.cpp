#include "Types.hpp"
#include "KittiLoader.hpp"
#include "RangeImageProjector.hpp"
#include "FrontEnd.hpp"
#include "PoseGraph.hpp"
#include "LoopClosureDetector.hpp"

#include <iostream>
#include <string>

int main() {
    // 1) Data loader
    slam::KittiLoader loader(
        "D:\\kitti_lidar\\00\\velodyne",
        "D:\\lidar_imu\\01\\imu"
    );

    // 2) Range image projector config
    slam::RangeImageConfig cfg;
    cfg.height = 64;
    cfg.width  = 1024;
    // cfg.v_fov_up / cfg.v_fov_down = defaults are okay for now

    slam::RangeImageProjector projector(cfg);

    // 3) Front-end (odometry)
    slam::FrontEnd frontend;

    // 4) Loop closure detector
    // Note: Update this path to your actual model file location
    std::string model_path = "src/pair_range_cnn_kitti_00_10.pt";
    slam::LoopClosureDetector loop_detector(
        model_path,
        0.5f,    // probability threshold
        10,      // keyframe interval (every 10 frames)
        50,      // min separation (at least 50 frames apart)
        20       // max candidates to check
    );

    // 5) Pose-graph back-end
    slam::PoseGraph graph;

    // Add first node (prior on initial pose)
    slam::Pose3D initial_pose;
    initial_pose.t = Eigen::Vector3d::Zero();
    initial_pose.q = Eigen::Quaterniond::Identity();
    int last_node_idx = graph.addNode(initial_pose);

    // 6) Main loop
    slam::LidarScan scan;
    int frame_count = 0;
    bool loop_closure_detected = false;

    while (loader.loadNextScan(scan)) {
        // Front-end: estimate relative motion (odometry)
        slam::PoseDelta odometry_delta = frontend.processFrame(scan);
        slam::Pose3D current_pose = frontend.currentPose();

        // Add new node to pose graph
        int curr_node_idx = graph.addNode(current_pose);

        // Add odometry edge from last to current
        graph.addOdometryEdge(
            last_node_idx,
            curr_node_idx,
            odometry_delta
        );

        // Loop closure detection (runs periodically on keyframes)
        std::vector<slam::LoopClosureCandidate> loop_candidates = 
            loop_detector.processFrame(scan, frame_count, curr_node_idx, current_pose, projector);

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

    std::cout << "Done. Total frames: " << frame_count 
              << ", Total nodes: " << graph.poses().size()
              << ", Total edges: " << graph.edges().size() << "\n";
    return 0;
}
