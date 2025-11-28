# LiDAR Range CNN - SLAM with Loop Closure Detection

A LiDAR-based SLAM system using range images and CNN-based loop closure detection.

## Dependencies

This project requires the following dependencies:

### Required Libraries

1. **Eigen3** (>= 3.3)
   - Linear algebra library
   - Install: `sudo apt-get install libeigen3-dev` (Linux) or use vcpkg/Conan

2. **PCL (Point Cloud Library)** (>= 1.8)
   - For ICP and point cloud processing
   - Install: `sudo apt-get install libpcl-dev` (Linux) or use vcpkg

3. **PyTorch/libtorch** (C++ API)
   - For neural network inference
   - Download from: https://pytorch.org/get-started/locally/
   - Extract and set `CMAKE_PREFIX_PATH` to the libtorch directory

4. **C++17 Compiler**
   - GCC >= 7, Clang >= 5, or MSVC >= 2017

### Data Requirements

- KITTI Odometry dataset (LiDAR scans in `.bin` format)
- KITTI IMU/OXTS data (in `.txt` format)
- Trained model file: `pair_range_cnn_kitti_00_10.pt` (should be in `src/` directory)

## Building

### Using CMake

```bash
mkdir build
cd build
cmake .. -DCMAKE_PREFIX_PATH=/path/to/libtorch
cmake --build . --config Release
```

### Windows (Visual Studio)

```powershell
mkdir build
cd build
cmake .. -DCMAKE_PREFIX_PATH=C:\path\to\libtorch -G "Visual Studio 16 2019" -A x64
cmake --build . --config Release
```

### Setting up libtorch

1. Download libtorch from https://pytorch.org/get-started/locally/
2. Extract it to a location (e.g., `C:\libtorch` or `/opt/libtorch`)
3. Set `CMAKE_PREFIX_PATH` when configuring:
   ```bash
   cmake .. -DCMAKE_PREFIX_PATH=/path/to/libtorch
   ```

## Running

The executable expects two command-line arguments:

```bash
./lidar_range_CNN <velodyne_dir> <imu_dir>
```

Example:
```bash
./lidar_range_CNN "D:\kitti_lidar\00\velodyne" "D:\lidar_imu\01\imu"
```

If no arguments are provided, it will use default paths (which may not exist).

## Project Structure

```
.
├── include/          # Header files
│   ├── Types.hpp
│   ├── FrontEnd.hpp
│   ├── KittiLoader.hpp
│   ├── LoopClosureDetector.hpp
│   ├── LoopClosureNet.hpp
│   ├── PoseGraph.hpp
│   └── RangeImageProjector.hpp
├── src/              # Source files
│   ├── main.cpp
│   ├── FrontEnd.cpp
│   ├── KittiLoader.cpp
│   ├── LoopClosureDetector.cpp
│   ├── LoopClosureNet.cpp
│   ├── PoseGraph.cpp
│   ├── RangeImageProjector.cpp
│   └── pair_range_cnn_kitti_00_10.pt  # Trained model
└── CMakeLists.txt
```

## Troubleshooting

### CMake can't find dependencies

- **Eigen3**: Install via package manager or set `EIGEN3_INCLUDE_DIR`
- **PCL**: Install via package manager or set `PCL_DIR`
- **libtorch**: Ensure `CMAKE_PREFIX_PATH` points to the libtorch root directory

### Runtime errors

- Check that data paths exist and contain `.bin` (LiDAR) and `.txt` (IMU) files
- Ensure the model file `pair_range_cnn_kitti_00_10.pt` exists in the build directory
- Verify that libtorch libraries are in the system PATH (Windows) or LD_LIBRARY_PATH (Linux)

## Notes

- The pose graph optimizer is currently a stub - you may want to integrate g2o or Ceres Solver for actual optimization
- The system processes frames sequentially and performs loop closure detection on keyframes
- Loop closures trigger pose graph optimization every 100 frames

