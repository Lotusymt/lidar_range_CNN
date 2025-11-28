#include "FrontEnd.hpp"

#include <iostream>                 // std::cerr for warnings
#include <pcl/point_cloud.h>        // pcl::PointCloud<pcl::PointXYZ>
#include <pcl/point_types.h>        // pcl::PointXYZ
#include <pcl/registration/icp.h>   // pcl::IterativeClosestPoint
#include <pcl/filters/voxel_grid.h> // pcl::VoxelGrid for downsampling

namespace {

    // Helper to build a PCL cloud from Eigen points and apply voxel downsampling.
    pcl::PointCloud<pcl::PointXYZ>::Ptr
    buildDownsampledCloud(const std::vector<Eigen::Vector3d>& points,
                          float voxel_leaf_size,
                          float max_range)
    {
        auto cloud = std::make_shared<pcl::PointCloud<pcl::PointXYZ>>();                  // create empty point cloud
    
        cloud->reserve(points.size());                                     // pre-allocate for speed
    
        for (const auto& p : points) {
            float x = static_cast<float>(p.x());                           // convert x to float
            float y = static_cast<float>(p.y());                           // convert y to float
            float z = static_cast<float>(p.z());                           // convert z to float
    
            float r2 = x * x + y * y + z * z;                              // squared distance from sensor
            if (max_range > 0.0f && r2 > max_range * max_range) {          // if beyond max_range, skip
                continue;                                                  // ignore far points
            }
    
            pcl::PointXYZ pt;
            pt.x = x;
            pt.y = y;
            pt.z = z;
            cloud->push_back(pt);                                          // add point to cloud
        }
    
        if (cloud->empty()) {                                              // if no points survived, return as-is
            return cloud;
        }
    
        auto filtered = pcl::PointCloud<pcl::PointXYZ>::Ptr(
            new pcl::PointCloud<pcl::PointXYZ>());                         // will hold downsampled cloud
    
        pcl::VoxelGrid<pcl::PointXYZ> voxel;
        voxel.setInputCloud(cloud);                                        // set input for voxel grid filter
        voxel.setLeafSize(voxel_leaf_size, voxel_leaf_size, voxel_leaf_size);
        // set voxel size in meters (x,y,z); larger -> fewer points, faster but coarser
    
        voxel.filter(*filtered);                                           // perform downsampling
    
        return filtered;                                                   // return downsampled cloud
    }
    
} // anonymous namespace
    

namespace slam {

FrontEnd::FrontEnd()
{
    // Start at origin with identity orientation.
    current_pose_.t = Eigen::Vector3d::Zero();
    current_pose_.q = Eigen::Quaterniond::Identity();
}

PoseDelta FrontEnd::processFrame(const LidarScan& scan)
{
    // First frame: no motion yet, just initialize state.
    if (!has_prev_scan_) {
        has_prev_scan_ = true;
        prev_scan_ = scan;  // store both points and imu

        PoseDelta delta;
        delta.T_prev_to_curr.t = Eigen::Vector3d::Zero();
        delta.T_prev_to_curr.q = Eigen::Quaterniond::Identity();
        delta.info = Eigen::Matrix<double, 6, 6>::Identity();
        // current_pose_ stays at origin.
        return delta;
    }

    // 1) IMU prior between previous and current scan (10 Hz OXTS -> single dt step).
    Pose3D imu_prior = integrateImuPrior(prev_scan_.imu, scan.imu);

    // 2) Refine with scan matching / ICP using IMU prior as initial guess.
    PoseDelta icp_delta = icpBetweenScans(prev_scan_, scan, imu_prior);

    // 3) Update global pose: T_world_curr = T_world_prev * T_prev_curr.
    const Pose3D& T = icp_delta.T_prev_to_curr;

    Pose3D new_pose;
    new_pose.q = current_pose_.q * T.q;                         // apply rotation delta
    new_pose.t = current_pose_.t + current_pose_.q * T.t;       // rotate & add translation

    current_pose_ = new_pose;
    prev_scan_ = scan;  // update previous scan + imu

    return icp_delta;
}

Pose3D FrontEnd::integrateImuPrior(const ImuSample& imu_prev,
                                   const ImuSample& imu_curr) const
{
    Pose3D delta;
    delta.t = Eigen::Vector3d::Zero();
    delta.q = Eigen::Quaterniond::Identity();

    // Single interval [t_prev, t_curr] because *_sync gives us 10 Hz OXTS.
    double dt = imu_curr.timestamp - imu_prev.timestamp;
    if (dt <= 0.0) {
        // If timestamps are weird, fall back to identity prior.
        return delta;
    }

    // Gyro integration: approximate average angular velocity over the interval.
    Eigen::Vector3d omega = 0.5 * (imu_prev.gyro + imu_curr.gyro);  // average gyro
    double angle = omega.norm() * dt;                                // total rotation angle

    if (angle > 1e-6) {
        Eigen::Vector3d axis = omega.normalized();                   // rotation axis
        Eigen::AngleAxisd aa(angle, axis);
        delta.q = Eigen::Quaterniond(aa);                            // relative rotation
    }

    // Translation prior stays zero; let ICP infer translation.
    return delta;
}

PoseDelta FrontEnd::icpBetweenScans(const LidarScan& prev,
    const LidarScan& curr,
    const Pose3D& initial_guess) const
{
    PoseDelta result;                                                  // will hold final relative pose + info

    // -----------------------------
    // 1) Build and downsample point clouds
    // -----------------------------

    const float voxel_leaf_size = 0.2f;                                // 0.2 m voxel grid
    const float max_range = 60.0f;                                     // ignore points beyond 60 m

    auto prev_cloud = buildDownsampledCloud(prev.points,
                voxel_leaf_size,
                max_range);                // downsampled previous scan
    auto curr_cloud = buildDownsampledCloud(curr.points,
                voxel_leaf_size,
                max_range);                // downsampled current scan

    if (prev_cloud->size() < 50 || curr_cloud->size() < 50) {          // if too few points, ICP is unreliable
        Pose3D delta;                                                  // fall back to IMU prior
        delta.q = initial_guess.q;
        delta.t = initial_guess.t;

        result.T_prev_to_curr = delta;
        result.info = Eigen::Matrix<double, 6, 6>::Identity();         // low-confidence identity info
        //add log warning
        std::cerr << "WARNING: Too few points, falling back to IMU prior" << std::endl;
        return result;                                                 // return fallback
    }

    // -----------------------------
    // 2) Build initial guess transform (Eigen Pose3D -> 4x4 matrix)
    // -----------------------------

    Eigen::Matrix4d guess_d = Eigen::Matrix4d::Identity();             // start with identity 4x4
    guess_d.block<3, 3>(0, 0) = initial_guess.q.toRotationMatrix();    // fill rotation block
    guess_d.block<3, 1>(0, 3) = initial_guess.t;                       // fill translation block

    Eigen::Matrix4f guess = guess_d.cast<float>();                     // cast to float for PCL

    // -----------------------------
    // 3) Configure and run ICP
    // -----------------------------

    pcl::IterativeClosestPoint<pcl::PointXYZ, pcl::PointXYZ> icp;      // ICP object
    icp.setInputSource(curr_cloud);                                    // "source" = current frame
    icp.setInputTarget(prev_cloud);                                    // "target" = previous frame

    icp.setMaxCorrespondenceDistance(30.0f);                           // max neighbor distance (meters)
    icp.setMaximumIterations(30);                                      // max ICP iterations
    icp.setTransformationEpsilon(1e-6);                                // stop if transform change is tiny
    icp.setEuclideanFitnessEpsilon(1e-5);                              // stop if error improvement is tiny

    pcl::PointCloud<pcl::PointXYZ> aligned;                            // will hold transformed current cloud

    icp.align(aligned, guess);                                         // run ICP with initial guess

    if (!icp.hasConverged()) {                                         // if ICP failed to converge
    std::cerr << "WARNING: ICP failed to converge, falling back to IMU prior" << std::endl;
    Pose3D delta;                                                  // fall back to IMU prior again
    delta.q = initial_guess.q;
    delta.t = initial_guess.t;

    result.T_prev_to_curr = delta;
    result.info = Eigen::Matrix<double, 6, 6>::Identity();
    return result;                                                 // return fallback
    }

    // -----------------------------
    // 4) Extract final transform and convert back to Pose3D
    // -----------------------------

    Eigen::Matrix4f T_icp_f = icp.getFinalTransformation();            // ICP result (float 4x4)
    Eigen::Matrix4d T_icp = T_icp_f.cast<double>();                    // convert to double for consistency

    Eigen::Matrix3d R = T_icp.block<3, 3>(0, 0);                       // rotation block
    Eigen::Vector3d t = T_icp.block<3, 1>(0, 3);                       // translation block

    Pose3D delta;
    delta.q = Eigen::Quaterniond(R);                                   // convert rotation matrix to quaternion
    delta.t = t;                                                       // set translation

    result.T_prev_to_curr = delta;                                     // store relative pose

    // -----------------------------
    // 5) Build an information matrix from ICP fitness (simple heuristic)
    // -----------------------------

    double fitness = icp.getFitnessScore();                            // mean squared error (lower = better)
    double weight = 1.0 / std::max(fitness, 1e-4);                     // invert error to get a weight

    Eigen::Matrix<double, 6, 6> info = Eigen::Matrix<double, 6, 6>::Zero();
    info(0, 0) = weight;                                               // rot x
    info(1, 1) = weight;                                               // rot y
    info(2, 2) = weight;                                               // rot z
    info(3, 3) = weight;                                               // trans x
    info(4, 4) = weight;                                               // trans y
    info(5, 5) = weight;                                               // trans z
    // simple isotropic info; you can tune rotation vs translation scaling later

    result.info = info;                                                // store information matrix

    return result;                                                     // return final ICP-based PoseDelta
}


}  // namespace slam
