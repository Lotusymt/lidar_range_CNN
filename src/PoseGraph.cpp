// PoseGraph.cpp
#include "PoseGraph.hpp"

#include <iostream>
#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <Eigen/SparseCholesky>
#include <cmath>

namespace slam {

int PoseGraph::addNode(const Pose3D& initial_estimate)
{
    poses_.push_back(initial_estimate);
    return static_cast<int>(poses_.size()) - 1;
}

void PoseGraph::addOdometryEdge(int from, int to, const PoseDelta& edge)
{
    Edge e;
    e.from = from;
    e.to   = to;
    e.meas = edge;
    edges_.push_back(e);
}

void PoseGraph::addLoopClosureEdge(int from, int to, const PoseDelta& edge)
{
    Edge e;
    e.from = from;
    e.to   = to;
    e.meas = edge;
    edges_.push_back(e);
    
    // Log loop closure detection
    std::cout << "[PoseGraph] Added loop closure edge: " << from 
              << " -> " << to << " (prob: " 
              << "N/A" << ")\n";
}

namespace {
    // Helper: Convert Pose3D to 4x4 transformation matrix
    Eigen::Matrix4d poseToMatrix(const Pose3D& pose)
    {
        Eigen::Matrix4d T = Eigen::Matrix4d::Identity();
        T.block<3, 3>(0, 0) = pose.q.toRotationMatrix();
        T.block<3, 1>(0, 3) = pose.t;
        return T;
    }

    // Helper: Convert 4x4 matrix to Pose3D
    Pose3D matrixToPose(const Eigen::Matrix4d& T)
    {
        Pose3D pose;
        pose.q = Eigen::Quaterniond(T.block<3, 3>(0, 0));
        pose.t = T.block<3, 1>(0, 3);
        return pose;
    }

    // Helper: Compute relative transform T_from^-1 * T_to
    Pose3D computeRelativeTransform(const Pose3D& from, const Pose3D& to)
    {
        Eigen::Matrix4d T_from = poseToMatrix(from);
        Eigen::Matrix4d T_to = poseToMatrix(to);
        Eigen::Matrix4d T_rel = T_from.inverse() * T_to;
        return matrixToPose(T_rel);
    }

    // Helper: Convert SE(3) error to 6D tangent space vector [rot, trans]
    Eigen::Vector6d poseError(const Pose3D& T_meas, const Pose3D& T_pred)
    {
        // Compute error: T_pred^-1 * T_meas
        Eigen::Matrix4d T_pred_mat = poseToMatrix(T_pred);
        Eigen::Matrix4d T_meas_mat = poseToMatrix(T_meas);
        Eigen::Matrix4d T_err = T_pred_mat.inverse() * T_meas_mat;

        // Extract rotation and translation error
        Eigen::Matrix3d R_err = T_err.block<3, 3>(0, 0);
        Eigen::Vector3d t_err = T_err.block<3, 1>(0, 3);

        // Convert rotation error to axis-angle (Lie algebra)
        Eigen::AngleAxisd angle_axis(R_err);
        Eigen::Vector3d rot_err = angle_axis.angle() * angle_axis.axis();

        Eigen::Vector6d error;
        error.head<3>() = rot_err;
        error.tail<3>() = t_err;
        return error;
    }

    // Helper: Apply Lie algebra update to pose: exp(se3) * T
    Pose3D applyUpdate(const Pose3D& pose, const Eigen::Vector6d& update)
    {
        // Extract rotation and translation updates
        Eigen::Vector3d rot_update = update.head<3>();
        Eigen::Vector3d trans_update = update.tail<3>();

        // Convert rotation update to quaternion
        double angle = rot_update.norm();
        Eigen::Quaterniond q_update;
        if (angle > 1e-8) {
            q_update = Eigen::AngleAxisd(angle, rot_update / angle);
        } else {
            q_update = Eigen::Quaterniond::Identity();
        }

        // Apply rotation update
        Eigen::Quaterniond q_new = pose.q * q_update;
        q_new.normalize();

        // Apply translation update (in the rotated frame)
        Eigen::Vector3d t_new = pose.t + pose.q.toRotationMatrix() * trans_update;

        Pose3D result;
        result.q = q_new;
        result.t = t_new;
        return result;
    }
}

void PoseGraph::optimize(int max_iterations)
{
    if (poses_.empty() || edges_.empty()) {
        std::cout << "[PoseGraph] Nothing to optimize: " 
                  << poses_.size() << " nodes, " 
                  << edges_.size() << " edges\n";
        return;
    }

    std::cout << "[PoseGraph] Starting optimization with "
              << poses_.size() << " nodes, "
              << edges_.size() << " edges, "
              << "max_iterations = " << max_iterations << "\n";

    const int num_nodes = static_cast<int>(poses_.size());
    const int state_dim = 6; // 6 DOF per pose (3 rot + 3 trans)

    // Convergence threshold
    const double convergence_threshold = 1e-6;
    const double lambda_init = 1e-4; // Levenberg-Marquardt damping

    for (int iter = 0; iter < max_iterations; ++iter) {
        // Build sparse linear system: H * delta = b
        // H is (num_nodes * 6) x (num_nodes * 6)
        // b is (num_nodes * 6) x 1

        using SparseMatrix = Eigen::SparseMatrix<double>;
        SparseMatrix H(num_nodes * state_dim, num_nodes * state_dim);
        Eigen::VectorXd b = Eigen::VectorXd::Zero(num_nodes * state_dim);

        std::vector<Eigen::Triplet<double>> triplets;
        triplets.reserve(edges_.size() * state_dim * state_dim * 4); // Estimate

        double total_error = 0.0;

        // Process each edge
        for (const auto& edge : edges_) {
            if (edge.from < 0 || edge.from >= num_nodes ||
                edge.to < 0 || edge.to >= num_nodes) {
                continue; // Skip invalid edges
            }

            // Get current pose estimates
            const Pose3D& pose_from = poses_[edge.from];
            const Pose3D& pose_to = poses_[edge.to];

            // Compute predicted relative transform
            Pose3D T_pred = computeRelativeTransform(pose_from, pose_to);

            // Get measured relative transform
            const Pose3D& T_meas = edge.meas.T_prev_to_curr;

            // Compute error in tangent space
            Eigen::Vector6d error = poseError(T_meas, T_pred);
            total_error += error.squaredNorm();

            // Get information matrix (inverse covariance)
            const Eigen::Matrix<double, 6, 6>& Omega = edge.meas.info;

            // Compute Jacobians using SE(3) adjoint representation
            // Error: log(T_pred^-1 * T_meas)
            // J_from = -Ad(T_pred^-1 * T_meas) * Ad(T_from^-1)
            // J_to = Ad(T_pred^-1)
            
            Eigen::Matrix4d T_from_mat = poseToMatrix(pose_from);
            Eigen::Matrix4d T_to_mat = poseToMatrix(pose_to);
            Eigen::Matrix4d T_pred_mat = poseToMatrix(T_pred);
            Eigen::Matrix4d T_meas_mat = poseToMatrix(T_meas);
            Eigen::Matrix4d T_err = T_pred_mat.inverse() * T_meas_mat;

            // SE(3) adjoint: Ad(T) = [R    0]
            //                        [t^R  R]
            // where t^ is the skew-symmetric matrix of t
            auto computeAdjoint = [](const Eigen::Matrix4d& T) -> Eigen::Matrix<double, 6, 6> {
                Eigen::Matrix3d R = T.block<3, 3>(0, 0);
                Eigen::Vector3d t = T.block<3, 1>(0, 3);
                
                // Skew-symmetric matrix of t: [t]×
                Eigen::Matrix3d t_skew;
                t_skew << 0, -t.z(), t.y(),
                         t.z(), 0, -t.x(),
                         -t.y(), t.x(), 0;
                
                Eigen::Matrix<double, 6, 6> Ad;
                Ad.block<3, 3>(0, 0) = R;
                Ad.block<3, 3>(0, 3) = Eigen::Matrix3d::Zero();
                Ad.block<3, 3>(3, 0) = t_skew * R;
                Ad.block<3, 3>(3, 3) = R;
                return Ad;
            };

            Eigen::Matrix<double, 6, 6> Ad_T_err = computeAdjoint(T_err);
            
            // Ad(T^-1) = Ad(T)^-1
            Eigen::Matrix4d T_from_inv = T_from_mat.inverse();
            Eigen::Matrix<double, 6, 6> Ad_T_from_inv = computeAdjoint(T_from_inv);
            
            Eigen::Matrix4d T_pred_inv = T_pred_mat.inverse();
            Eigen::Matrix<double, 6, 6> Ad_T_pred_inv = computeAdjoint(T_pred_inv);

            // Jacobians for error = log(T_pred^-1 * T_meas)
            // d/dT_from: -Ad(T_err) * Ad(T_from^-1)
            // d/dT_to: Ad(T_pred^-1)
            Eigen::Matrix<double, 6, 6> J_from = -Ad_T_err * Ad_T_from_inv;
            Eigen::Matrix<double, 6, 6> J_to = Ad_T_pred_inv;

            // Add to Hessian: J^T * Omega * J
            Eigen::Matrix<double, 6, 6> H_from_from = J_from.transpose() * Omega * J_from;
            Eigen::Matrix<double, 6, 6> H_from_to = J_from.transpose() * Omega * J_to;
            Eigen::Matrix<double, 6, 6> H_to_from = J_to.transpose() * Omega * J_from;
            Eigen::Matrix<double, 6, 6> H_to_to = J_to.transpose() * Omega * J_to;

            // Add to right-hand side: J^T * Omega * error
            Eigen::Vector6d b_from = J_from.transpose() * Omega * error;
            Eigen::Vector6d b_to = J_to.transpose() * Omega * error;

            // Add blocks to sparse matrix
            for (int i = 0; i < state_dim; ++i) {
                for (int j = 0; j < state_dim; ++j) {
                    // from-from block
                    triplets.emplace_back(
                        edge.from * state_dim + i,
                        edge.from * state_dim + j,
                        H_from_from(i, j));
                    // from-to block
                    triplets.emplace_back(
                        edge.from * state_dim + i,
                        edge.to * state_dim + j,
                        H_from_to(i, j));
                    // to-from block
                    triplets.emplace_back(
                        edge.to * state_dim + i,
                        edge.from * state_dim + j,
                        H_to_from(i, j));
                    // to-to block
                    triplets.emplace_back(
                        edge.to * state_dim + i,
                        edge.to * state_dim + j,
                        H_to_to(i, j));
                }
            }

            // Add to right-hand side
            b.segment<6>(edge.from * state_dim) += b_from;
            b.segment<6>(edge.to * state_dim) += b_to;
        }

        // Fix first pose (anchor node) to remove gauge freedom
        // Add large diagonal to first pose's block
        for (int i = 0; i < state_dim; ++i) {
            triplets.emplace_back(i, i, 1e6);
        }

        // Build sparse matrix
        H.setFromTriplets(triplets.begin(), triplets.end());

        // Add Levenberg-Marquardt damping
        for (int i = 0; i < num_nodes * state_dim; ++i) {
            H.coeffRef(i, i) += lambda_init;
        }

        // Solve linear system: H * delta = -b
        Eigen::SimplicialLDLT<SparseMatrix> solver;
        solver.compute(H);
        
        if (solver.info() != Eigen::Success) {
            std::cerr << "[PoseGraph] Warning: Linear solver failed at iteration " 
                      << iter << "\n";
            break;
        }

        Eigen::VectorXd delta = solver.solve(-b);

        if (solver.info() != Eigen::Success) {
            std::cerr << "[PoseGraph] Warning: Linear solve failed at iteration " 
                      << iter << "\n";
            break;
        }

        // Apply updates to poses
        double max_update = 0.0;
        for (int i = 0; i < num_nodes; ++i) {
            Eigen::Vector6d update = delta.segment<6>(i * state_dim);
            max_update = std::max(max_update, update.norm());
            poses_[i] = applyUpdate(poses_[i], update);
        }

        // Print progress
        std::cout << "[PoseGraph] Iteration " << iter + 1 
                  << ": error = " << total_error 
                  << ", max_update = " << max_update << "\n";

        // Check convergence
        if (max_update < convergence_threshold) {
            std::cout << "[PoseGraph] Converged after " << iter + 1 << " iterations\n";
            break;
        }
    }

    std::cout << "[PoseGraph] Optimization complete\n";
}

} // namespace slam
