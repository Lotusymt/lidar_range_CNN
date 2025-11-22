import os
import zipfile

# ---------- CONFIGURE THESE PATHS ----------
ZIP_ROOT     = r"E:\\"           # folder where your *_sync.zip files are stored
# example: E:\2011_10_03_drive_0027_sync.zip

RAW_ROOT     = r"E:\kitti_raw"  # where we will unzip the raw *_sync folders

ODOM_ROOT    = r"D:\kitti_lidar"  # odometry lidar root
# expected layout:
# D:\kitti_lidar\00\velodyne\000000.bin
# D:\kitti_lidar\01\velodyne\000000.bin
# ...

IMU_OUT_ROOT = r"D:\lidar_imu"    # where we will write simplified IMU files
# target layout:
# D:\lidar_imu\00\imu\000000.txt
# D:\lidar_imu\01\imu\000000.txt
# each txt: "timestamp ax ay az wx wy wz"


# Mapping odom seq (00-10) -> raw drive + start offset (end is sanity check)
# We skip 03 because its raw drive (2011_09_26_drive_0067) is not available.
SEQUENCE_MAPPING = {
    0: {"raw": "2011_10_03_drive_0027", "start": 0,    "end": 4540},
    1: {"raw": "2011_10_03_drive_0042", "start": 0,    "end": 1100},
    2: {"raw": "2011_10_03_drive_0034", "start": 0,    "end": 4660},
    # 3 is missing, do NOT use
    4: {"raw": "2011_09_30_drive_0016", "start": 0,    "end": 270},
    5: {"raw": "2011_09_30_drive_0018", "start": 0,    "end": 2760},
    6: {"raw": "2011_09_30_drive_0020", "start": 0,    "end": 1100},
    7: {"raw": "2011_09_30_drive_0027", "start": 0,    "end": 1100},
    8: {"raw": "2011_09_30_drive_0028", "start": 1100, "end": 5170},
    9: {"raw": "2011_09_30_drive_0033", "start": 0,    "end": 1590},
    10: {"raw": "2011_09_30_drive_0034", "start": 0,   "end": 1200},
}


def ensure_unzipped(raw_name: str):
    # Make sure RAW_ROOT exists
    os.makedirs(RAW_ROOT, exist_ok=True)

    dst_dir = os.path.join(RAW_ROOT, f"{raw_name}_sync")
    # If folder already exists, we assume it's correctly unzipped
    if os.path.isdir(dst_dir):
        print(f"[INFO] {dst_dir} already exists, skipping unzip.")
        return

    # Look for the zip in ZIP_ROOT
    zip_name = f"{raw_name}_sync.zip"
    zip_path = os.path.join(ZIP_ROOT, zip_name)

    if not os.path.exists(zip_path):
        print(f"[WARN] Zip not found: {zip_path}")
        print("       Please put the *_sync.zip files into ZIP_ROOT.")
        return

    print(f"[INFO] Unzipping {zip_path} -> {RAW_ROOT}")
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(RAW_ROOT)

    if not os.path.isdir(dst_dir):
        print(f"[WARN] After unzip, directory not found: {dst_dir}")
    else:
        print(f"[INFO] Unzipped to {dst_dir}")


def parse_oxts_line(line: str):
    # OXTS line is space-separated floats.
    # Known KITTI format indices:
    #  0 lat, 1 lon, 2 alt,
    #  3 roll, 4 pitch, 5 yaw,
    #  6 vn, 7 ve, 8 vf, 9 vl, 10 vu,
    # 11 ax, 12 ay, 13 az,
    # 17 wx, 18 wy, 19 wz, ...
    vals = [float(x) for x in line.strip().split()]
    ax, ay, az = vals[11], vals[12], vals[13]
    wx, wy, wz = vals[17], vals[18], vals[19]
    return ax, ay, az, wx, wy, wz


def generate_imu_for_sequence(seq_id: int):
    info = SEQUENCE_MAPPING[seq_id]
    raw_name = info["raw"]
    start = info["start"]
    end = info["end"]

    seq_str = f"{seq_id:02d}"

    # Ensure raw drive is unzipped
    ensure_unzipped(raw_name)

    # Lidar odometry folder for this seq: D:\kitti_lidar\00\velodyne
    velo_dir = os.path.join(ODOM_ROOT, seq_str, "velodyne")
    if not os.path.isdir(velo_dir):
        print(f"[WARN] Seq {seq_str}: velodyne dir not found: {velo_dir}")
        return

    # OXTS folder inside raw: E:\kitti_raw\2011_xx_xx_drive_xxxx_sync\oxts\data
    raw_oxts_dir = os.path.join(RAW_ROOT, f"{raw_name}_sync", "oxts", "data")
    if not os.path.isdir(raw_oxts_dir):
        print(f"[WARN] Seq {seq_str}: oxts data dir not found: {raw_oxts_dir}")
        return

    # Output IMU folder: D:\lidar_imu\00\imu
    imu_out_dir = os.path.join(IMU_OUT_ROOT, seq_str, "imu")
    os.makedirs(imu_out_dir, exist_ok=True)

    # List lidar frames in odometry dataset
    bin_files = sorted(f for f in os.listdir(velo_dir) if f.endswith(".bin"))
    print(f"[INFO] Seq {seq_str}: {len(bin_files)} velodyne frames.")

    for bin_name in bin_files:
        stem = os.path.splitext(bin_name)[0]  # "000123" -> frame index
        try:
            frame_idx = int(stem)
        except ValueError:
            print(f"[WARN] Unexpected file in {velo_dir}: {bin_name}, skipping.")
            continue

        raw_idx = start + frame_idx
        if raw_idx < start or raw_idx > end:
            print(
                f"[WARN] frame {frame_idx:010d}: raw_idx {raw_idx} out of range "
                f"[{start}, {end}] for seq {seq_str}, skipping."
            )
            continue

        oxts_path = os.path.join(raw_oxts_dir, f"{raw_idx:010d}.txt")
        if not os.path.exists(oxts_path):
            print(f"[WARN] OXTS file not found: {oxts_path}, skipping.")
            continue

        # Read the OXTS line
        with open(oxts_path, "r") as f:
            line = f.readline()
        ax, ay, az, wx, wy, wz = parse_oxts_line(line)

        # Simple synthetic timestamp: 0.1 seconds per frame
        timestamp = frame_idx * 0.1

        imu_out_path = os.path.join(imu_out_dir, f"{frame_idx:010d}.txt")
        # Write single line: timestamp ax ay az wx wy wz
        with open(imu_out_path, "w") as f:
            f.write(
                f"{timestamp:.6f} {ax:.9f} {ay:.9f} {az:.9f} "
                f"{wx:.9f} {wy:.9f} {wz:.9f}\n"
            )

    print(f"[INFO] Done IMU export for seq {seq_str} -> {imu_out_dir}")


def main():
    # sequences that have raw data (skip 3)
    seq_ids = [0, 1, 2, 4, 5, 6, 7, 8, 9, 10]
    for seq_id in seq_ids:
        generate_imu_for_sequence(seq_id)


if __name__ == "__main__":
    main()
