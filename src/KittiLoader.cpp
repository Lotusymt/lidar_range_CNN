// KittiLoader.cpp
#include "KittiLoader.hpp"

#include <filesystem>
#include <fstream>
#include <iostream>
#include <sstream>
#include <algorithm>

namespace slam {

namespace fs = std::filesystem;

KittiLoader::KittiLoader(const std::string& velodyne_dir,
                         const std::string& imu_dir)
{
    // Collect LiDAR .bin files
    try {
        for (const auto& entry : fs::directory_iterator(velodyne_dir)) {
            if (!entry.is_regular_file()) continue;
            if (entry.path().extension() == ".bin") {
                lidar_files_.push_back(entry.path().string());
            }
        }
    } catch (const std::exception& e) {
        std::cerr << "[KittiLoader] Error reading velodyne dir: "
                  << velodyne_dir << " : " << e.what() << "\n";
    }

    // Collect IMU .txt files
    try {
        for (const auto& entry : fs::directory_iterator(imu_dir)) {
            if (!entry.is_regular_file()) continue;
            if (entry.path().extension() == ".txt") {
                imu_files_.push_back(entry.path().string());
            }
        }
    } catch (const std::exception& e) {
        std::cerr << "[KittiLoader] Error reading imu dir: "
                  << imu_dir << " : " << e.what() << "\n";
    }

    // Sort lexicographically so 000000.bin, 000001.bin, ... are in order.
    auto lexicographic_sort = [](std::vector<std::string>& v) {
        std::sort(v.begin(), v.end(),
                  [](const std::string& a, const std::string& b) {
                      return a < b;
                  });
    };

    lexicographic_sort(lidar_files_);
    lexicographic_sort(imu_files_);

    size_ = std::min(lidar_files_.size(), imu_files_.size());

    std::cout << "[KittiLoader] Found " << lidar_files_.size()
              << " lidar scans, " << imu_files_.size()
              << " imu files, using " << size_ << " matched frames.\n";
}

bool KittiLoader::loadNextScan(LidarScan& scan)
{
    if (index_ >= size_) {
        return false;  // no more frames
    }

    const std::string& lidar_path = lidar_files_[index_];
    const std::string& imu_path   = imu_files_[index_];

    // 1) Load LiDAR points
    scan.points.clear();
    if (!loadLidarBin(lidar_path, scan.points)) {
        std::cerr << "[KittiLoader] Failed to load lidar: "
                  << lidar_path << "\n";
        return false;
    }

    // 2) Load IMU sample
    if (!loadImuTxt(imu_path, scan.imu)) {
        std::cerr << "[KittiLoader] Failed to load imu: "
                  << imu_path << "\n";
        return false;
    }

    // 3) Timestamp
    // If you have real timestamps elsewhere, set them here instead.
    // For now, assume ~10 Hz and use index * 0.1s as a dummy value.
    scan.timestamp = static_cast<double>(index_) * 0.1;

    ++index_;
    return true;
}

void KittiLoader::reset()
{
    index_ = 0;
}

bool KittiLoader::loadLidarBin(const std::string& path,
                               std::vector<Eigen::Vector3d>& points)
{
    std::ifstream ifs(path, std::ios::binary);
    if (!ifs) {
        std::cerr << "[KittiLoader] Cannot open lidar file: "
                  << path << "\n";
        return false;
    }

    // Determine file size
    ifs.seekg(0, std::ios::end);
    std::streamsize size = ifs.tellg();
    ifs.seekg(0, std::ios::beg);

    // KITTI velodyne: [x y z intensity] as float32
    constexpr std::size_t FLOATS_PER_POINT = 4;
    if (size % (FLOATS_PER_POINT * sizeof(float)) != 0) {
        std::cerr << "[KittiLoader] Unexpected lidar file size: "
                  << path << "\n";
        return false;
    }

    const std::size_t num_floats  = static_cast<std::size_t>(size) / sizeof(float);
    const std::size_t num_points  = num_floats / FLOATS_PER_POINT;

    std::vector<float> buffer(num_floats);
    if (!ifs.read(reinterpret_cast<char*>(buffer.data()), size)) {
        std::cerr << "[KittiLoader] Failed to read lidar file: "
                  << path << "\n";
        return false;
    }

    points.resize(num_points);

    for (std::size_t i = 0; i < num_points; ++i) {
        float x = buffer[FLOATS_PER_POINT * i + 0];
        float y = buffer[FLOATS_PER_POINT * i + 1];
        float z = buffer[FLOATS_PER_POINT * i + 2];
        // float intensity = buffer[FLOATS_PER_POINT * i + 3]; // not used here

        points[i] = Eigen::Vector3d(
            static_cast<double>(x),
            static_cast<double>(y),
            static_cast<double>(z)
        );
    }

    return true;
}

bool KittiLoader::loadImuTxt(const std::string& path,
                             ImuSample& imu)
{
    std::ifstream ifs(path);
    if (!ifs) {
        std::cerr << "[KittiLoader] Cannot open imu file: "
                  << path << "\n";
        return false;
    }

    // Assumes one line like:
    //   timestamp ax ay az gx gy gz
    // Adapt here if your format is different.
    std::string line;
    while (std::getline(ifs, line)) {
        if (!line.empty()) break;
    }

    if (line.empty()) {
        std::cerr << "[KittiLoader] Empty imu file: "
                  << path << "\n";
        return false;
    }

    std::istringstream iss(line);
    double t, ax, ay, az, gx, gy, gz;
    if (!(iss >> t >> ax >> ay >> az >> gx >> gy >> gz)) {
        std::cerr << "[KittiLoader] Failed to parse imu line in: "
                  << path << "\n";
        return false;
    }

    imu.timestamp = t;
    imu.accel = Eigen::Vector3d(ax, ay, az);
    imu.gyro  = Eigen::Vector3d(gx, gy, gz);

    return true;
}

} // namespace slam
