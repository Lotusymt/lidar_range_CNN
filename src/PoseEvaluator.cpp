#include "PoseEvaluator.hpp"
#include <fstream>
#include <sstream>
#include <iostream>
#include <cmath>
#include <algorithm>

namespace slam {

std::vector<Pose3D> PoseEvaluator::loadGroundTruth(const std::string& filename)
{
    std::vector<Pose3D> poses;
    std::ifstream file(filename);
    
    if (!file.is_open()) {
        throw std::runtime_error("PoseEvaluator: Cannot open ground truth file: " + filename);
    }

    std::string line;
    while (std::getline(file, line)) {
        std::istringstream iss(line);
        std::vector<double> values;
        double val;
        
        while (iss >> val) {
            values.push_back(val);
        }

        if (values.size() != 12) {
            continue;  // Skip malformed lines
        }

        // KITTI format: 3x4 matrix (R|t) stored row-major
        // Extract rotation matrix (3x3) and translation (3x1)
        Eigen::Matrix3d R;
        R << values[0], values[1], values[2],
             values[4], values[5], values[6],
             values[8], values[9], values[10];

        Eigen::Vector3d t(values[3], values[7], values[11]);

        Pose3D pose;
        pose.t = t;
        pose.q = Eigen::Quaterniond(R);

        poses.push_back(pose);
    }

    file.close();
    return poses;
}

double PoseEvaluator::computeATE(const std::vector<Pose3D>& estimated,
                                  const std::vector<Pose3D>& ground_truth)
{
    if (estimated.size() != ground_truth.size()) {
        throw std::runtime_error("PoseEvaluator: Estimated and ground truth sizes don't match");
    }

    if (estimated.empty()) {
        return 0.0;
    }

    // Compute translation errors
    double sum_squared_error = 0.0;
    for (size_t i = 0; i < estimated.size(); ++i) {
        Eigen::Vector3d error = estimated[i].t - ground_truth[i].t;
        sum_squared_error += error.squaredNorm();
    }

    return std::sqrt(sum_squared_error / estimated.size());
}

double PoseEvaluator::computeRPE(const std::vector<Pose3D>& estimated,
                                 const std::vector<Pose3D>& ground_truth,
                                 int delta)
{
    if (estimated.size() != ground_truth.size()) {
        throw std::runtime_error("PoseEvaluator: Estimated and ground truth sizes don't match");
    }

    if (static_cast<size_t>(delta) >= estimated.size()) {
        return 0.0;
    }

    // Compute relative pose errors
    std::vector<double> errors;
    errors.reserve(estimated.size() - delta);

    for (size_t i = 0; i < estimated.size() - delta; ++i) {
        // Relative transform for estimated: T_i_to_i+delta
        Eigen::Quaterniond q_est_rel = estimated[i].q.conjugate() * estimated[i + delta].q;
        Eigen::Vector3d t_est_rel = estimated[i].q.conjugate() * (estimated[i + delta].t - estimated[i].t);

        // Relative transform for ground truth: T_i_to_i+delta
        Eigen::Quaterniond q_gt_rel = ground_truth[i].q.conjugate() * ground_truth[i + delta].q;
        Eigen::Vector3d t_gt_rel = ground_truth[i].q.conjugate() * (ground_truth[i + delta].t - ground_truth[i].t);

        // Error in relative translation
        Eigen::Vector3d error = t_est_rel - t_gt_rel;
        errors.push_back(error.norm());
    }

    if (errors.empty()) {
        return 0.0;
    }

    // Compute RMSE
    double sum_squared = 0.0;
    for (double err : errors) {
        sum_squared += err * err;
    }

    return std::sqrt(sum_squared / errors.size());
}

void PoseEvaluator::evaluate(const std::vector<Pose3D>& estimated,
                             const std::string& ground_truth_file)
{
    try {
        std::vector<Pose3D> ground_truth = loadGroundTruth(ground_truth_file);

        if (estimated.size() != ground_truth.size()) {
            std::cerr << "Warning: Estimated poses (" << estimated.size() 
                      << ") != Ground truth poses (" << ground_truth.size() << ")\n";
            std::cerr << "Evaluating using minimum size: " 
                      << std::min(estimated.size(), ground_truth.size()) << "\n";
            
            // Truncate to minimum size
            size_t min_size = std::min(estimated.size(), ground_truth.size());
            std::vector<Pose3D> est_trunc(estimated.begin(), estimated.begin() + min_size);
            std::vector<Pose3D> gt_trunc(ground_truth.begin(), ground_truth.begin() + min_size);
            
            double ate = computeATE(est_trunc, gt_trunc);
            double rpe = computeRPE(est_trunc, gt_trunc);
            
            std::cout << "\n=== Pose Evaluation Results ===\n";
            std::cout << "ATE (Absolute Trajectory Error): " << ate << " m (RMSE)\n";
            std::cout << "RPE (Relative Pose Error): " << rpe << " m (RMSE)\n";
            std::cout << "================================\n";
        } else {
            double ate = computeATE(estimated, ground_truth);
            double rpe = computeRPE(estimated, ground_truth);
            
            std::cout << "\n=== Pose Evaluation Results ===\n";
            std::cout << "ATE (Absolute Trajectory Error): " << ate << " m (RMSE)\n";
            std::cout << "RPE (Relative Pose Error): " << rpe << " m (RMSE)\n";
            std::cout << "================================\n";
        }
    } catch (const std::exception& e) {
        std::cerr << "Error evaluating poses: " << e.what() << "\n";
        std::cerr << "Skipping evaluation.\n";
    }
}

} // namespace slam

