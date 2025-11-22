// PoseGraph.hpp
#pragma once

#include "Types.hpp"

#include <vector>

namespace slam {

/**
 * @brief Simple pose-graph structure for SE(3) SLAM.
 *
 * For now this is just storage + a stub optimize() method.
 */
class PoseGraph {
public:
    struct Edge {
        int from;      ///< index of previous node
        int to;        ///< index of current node
        PoseDelta meas;///< relative measurement from 'from' to 'to'
    };

    /// @brief Add a new node (pose) to the graph. Returns its index.
    int addNode(const Pose3D& initial_estimate);

    /// @brief Add an odometry edge between two nodes.
    void addOdometryEdge(int from, int to, const PoseDelta& edge);

    /// @brief Add a loop closure edge between two nodes.
    /// 
    /// Loop closure edges typically have higher information (confidence)
    /// than odometry edges since they represent strong constraints.
    /// 
    /// @param from Node index of the past keyframe
    /// @param to Node index of the current keyframe
    /// @param edge Relative transform from 'from' to 'to' with information matrix
    void addLoopClosureEdge(int from, int to, const PoseDelta& edge);

    /// @brief Run graph optimization (stub for now).
    void optimize(int max_iterations = 10);

    /// @brief Access the poses.
    const std::vector<Pose3D>& poses() const { return poses_; }

    /// @brief Access the edges.
    const std::vector<Edge>& edges() const { return edges_; }

private:
    std::vector<Pose3D> poses_;
    std::vector<Edge>   edges_;
};

} // namespace slam
