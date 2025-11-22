// PoseGraph.cpp
#include "PoseGraph.hpp"

#include <iostream>

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

void PoseGraph::optimize(int max_iterations)
{
    // TODO: connect to real optimizer (g2o / Ceres / etc.)
    // For now, we just print some info so you can see it is called.

    std::cout << "[PoseGraph] optimize() called with "
              << poses_.size() << " nodes, "
              << edges_.size() << " edges, "
              << "max_iterations = " << max_iterations << "\n";

    // No-op optimization: we keep poses as they are.
}

} // namespace slam
