#!/usr/bin/env python3
"""
Python wrapper for the SLAM C++ executable.
Allows easy parameter changes without recompiling.
"""

import os
import sys
import subprocess
from pathlib import Path

# Default configuration - modify these as needed
CONFIG = {
    "velodyne_dir": r"D:\odom_dataset\kitti_lidar\00\velodyne",
    "imu_dir": r"D:\odom_dataset\lidar_imu\00\imu",
    "model_path": "src/pair_range_cnn_kitti_00_10.pt",
    "executable": "./build/bin/lidar_range_CNN.exe" if sys.platform == "win32" else "./build/bin/lidar_range_CNN",
}

def main():
    """Run the SLAM executable with configurable parameters."""
    
    # Parse command-line arguments if provided
    if len(sys.argv) >= 3:
        CONFIG["velodyne_dir"] = sys.argv[1]
        CONFIG["imu_dir"] = sys.argv[2]
    elif len(sys.argv) == 2:
        print(f"Usage: {sys.argv[0]} [velodyne_dir] [imu_dir]")
        print(f"Using default paths from CONFIG")
    
    # Check if executable exists (try Debug first, then Release, then direct path)
    exe_path = Path(CONFIG["executable"])
    
    # If direct path doesn't exist, try Release first (matches libtorch), then Debug
    if not exe_path.exists():
        # Try Release build first (matches Release libtorch package)
        release_path = Path("./build/bin/Release/lidar_range_CNN.exe" if sys.platform == "win32" else "./build/bin/Release/lidar_range_CNN")
        if release_path.exists():
            exe_path = release_path
            print(f"Found Release build at {exe_path}")
        else:
            # Try Debug build (may have ABI mismatch issues with Release libtorch)
            debug_path = Path("./build/bin/Debug/lidar_range_CNN.exe" if sys.platform == "win32" else "./build/bin/Debug/lidar_range_CNN")
            if debug_path.exists():
                exe_path = debug_path
                print(f"Found Debug build at {exe_path} (WARNING: May have issues with Release libtorch)")
            else:
                print(f"Error: Executable not found at {CONFIG['executable']}")
                print(f"  Also checked: {debug_path}")
                print(f"  Also checked: {release_path}")
                print(f"Please build the project first or update CONFIG['executable']")
                return 1
    
    # Build command
    cmd = [
        str(exe_path),
        CONFIG["velodyne_dir"],
        CONFIG["imu_dir"]
    ]
    
    # Set up environment with DLL paths
    env = os.environ.copy()
    
    # Add the Debug/Release directory to PATH (where DLLs are located)
    exe_dir = exe_path.parent
    current_path = env.get("PATH", "")
    if str(exe_dir) not in current_path:
        env["PATH"] = str(exe_dir) + os.pathsep + current_path
    
    # Try to find and add libtorch DLL directory
    # Common locations for libtorch DLLs
    libtorch_paths = [
        os.path.join(os.path.dirname(exe_dir), "..", "libtorch", "lib"),
        "D:/lib/libtorch/lib",
        "C:/libtorch/lib",
        os.path.expanduser("~/libtorch/lib"),
    ]
    
    libtorch_added = False
    for libtorch_path in libtorch_paths:
        abs_path = os.path.abspath(libtorch_path)
        if os.path.exists(abs_path) and os.path.isdir(abs_path):
            if abs_path not in env["PATH"]:
                env["PATH"] = abs_path + os.pathsep + env["PATH"]
                print(f"Added libtorch DLL path: {abs_path}")
                libtorch_added = True
            break
    
    # Try to find and add PCL DLL directory (if PCL is used)
    pcl_paths = [
        "D:/lib/pcl/bin",
        "D:/lib/pcl/lib",
        "C:/PCL/bin",
        "C:/PCL/lib",
    ]
    
    for pcl_path in pcl_paths:
        abs_path = os.path.abspath(pcl_path)
        if os.path.exists(abs_path) and os.path.isdir(abs_path):
            if abs_path not in env["PATH"]:
                env["PATH"] = abs_path + os.pathsep + env["PATH"]
                print(f"Added PCL DLL path: {abs_path}")
            break
    
    print("=" * 60)
    print("Running SLAM with configuration:")
    print(f"  Velodyne dir: {CONFIG['velodyne_dir']}")
    print(f"  IMU dir:      {CONFIG['imu_dir']}")
    print(f"  Executable:   {exe_path}")
    print(f"  DLL path:     {exe_dir}")
    print("=" * 60)
    print()
    
    # Run the executable
    try:
        result = subprocess.run(cmd, check=True, env=env)
        return result.returncode
    except subprocess.CalledProcessError as e:
        print(f"Error: Executable failed with return code {e.returncode}")
        return e.returncode
    except FileNotFoundError:
        print(f"Error: Executable not found at {exe_path}")
        return 1

if __name__ == "__main__":
    sys.exit(main())

