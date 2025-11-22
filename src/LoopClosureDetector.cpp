#include "slam/LoopClosureDetector.hpp"
#include <algorithm>
#include <cmath>

namespace slam {

LoopClosureDetector::LoopClosureDetector(const std::string& model_path,
                                         float probability_threshold,
                                         int keyframe_interval,
                                         int min_separation,
                                         int max_candidates)
    : net_(std::make_unique<LoopClosureNet>(model_path))
    , probability_threshold_(probability_threshold)
    , keyframe_interval_(keyframe_interval)
    , min_separation_(min_separation)
    , max_candidates_(max_candidates)
{
}

std::vector<LoopClosureCandidate> LoopClosureDetector::processFrame(
    const LidarScan& scan,
    int current_frame_id,
    int current_node_index,
    const Pose3D& current_pose,
    const RangeImageProjector& projector)
{
    std::vector<LoopClosureCandidate> candidates;

    // Project current scan to range image
    RangeImage current_range_img;
    projector.project(scan, current_range_img);
    current_range_img.timestamp = scan.timestamp;

    // Decide if this should be a keyframe
    bool is_keyframe = (current_frame_id % keyframe_interval_ == 0) || 
                       (keyframes_.empty());

    if (!is_keyframe) {
        // Not a keyframe, no loop closure detection
        return candidates;
    }

    // This is a keyframe - check against past keyframes
    if (keyframes_.size() < 2) {
        // Not enough keyframes yet, just store this one
        Keyframe kf;
        kf.frame_id = current_frame_id;
        kf.node_index = current_node_index;
        kf.range_image = current_range_img;
        kf.pose = current_pose;
        keyframes_.push_back(kf);
        return candidates;
    }

    // Check against past keyframes (from oldest to newest, but skip very recent ones)
    int num_to_check = std::min(static_cast<int>(keyframes_.size()), max_candidates_);
    
    // Start checking from older keyframes (better loop closure candidates)
    // Skip the most recent keyframes (min_separation constraint)
    int start_idx = 0;
    if (static_cast<int>(keyframes_.size()) > min_separation_ / keyframe_interval_) {
        // Skip recent keyframes
        start_idx = static_cast<int>(keyframes_.size()) - 
                    (min_separation_ / keyframe_interval_) - num_to_check;
        start_idx = std::max(0, start_idx);
    }

    float best_prob = 0.0f;
    int best_keyframe_idx = -1;

    for (int i = start_idx; i < static_cast<int>(keyframes_.size()); ++i) {
        const auto& past_kf = keyframes_[i];
        
        // Skip if too close in frame space
        if (current_frame_id - past_kf.frame_id < min_separation_) {
            continue;
        }

        // Run neural network to get loop closure probability
        float prob = net_->loopClosureProbability(current_range_img, past_kf.range_image);

        if (prob > probability_threshold_ && prob > best_prob) {
            best_prob = prob;
            best_keyframe_idx = i;
        }
    }

    // If we found a good candidate, create a LoopClosureCandidate
    if (best_keyframe_idx >= 0) {
        LoopClosureCandidate candidate;
        candidate.keyframe_id = keyframes_[best_keyframe_idx].node_index;  // Use node index, not frame_id
        candidate.probability = best_prob;
        
        // Compute relative transform from past to current keyframe
        // PoseDelta expects T_prev_to_curr, so we need T_past_to_curr
        const Pose3D& past_pose = keyframes_[best_keyframe_idx].pose;
        
        // T_past_to_curr = T_world_to_curr * T_past_to_world
        // T_past_to_world = inv(T_world_to_past)
        Eigen::Quaterniond q_past_to_world = past_pose.q.conjugate();
        Eigen::Vector3d t_past_to_world = -q_past_to_world * past_pose.t;
        
        // T_world_to_curr = current_pose
        // T_past_to_curr = T_world_to_curr * T_past_to_world
        candidate.T_past_to_curr.q = current_pose.q * q_past_to_world;
        candidate.T_past_to_curr.t = current_pose.q * t_past_to_world + current_pose.t;
        
        candidates.push_back(candidate);
    }

    // Store this keyframe for future comparisons
    Keyframe kf;
    kf.frame_id = current_frame_id;
    kf.node_index = current_node_index;  // Use the node index from pose graph
    kf.range_image = current_range_img;
    kf.pose = current_pose;
    keyframes_.push_back(kf);

    return candidates;
}

int LoopClosureDetector::getKeyframeNodeIndex(int frame_id) const
{
    for (const auto& kf : keyframes_) {
        if (kf.frame_id == frame_id) {
            return kf.node_index;
        }
    }
    return -1;
}

} // namespace slam

