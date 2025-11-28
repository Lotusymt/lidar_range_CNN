import os
import glob
import numpy as np
import torch
from torch.utils.data import Dataset
from range_kitti import read_velodyne_bin, cloud_to_range_image

# Configuration: Set via environment variable or use default
KITTI_LIDAR_ROOT = os.getenv("KITTI_LIDAR_ROOT", r"D:\kitti_lidar")
KITTI_POSES_ROOT = os.getenv("KITTI_POSES_ROOT", r"D:\data_odometry_poses\dataset")


class KittiRangeDataset(Dataset):
    """
    Returns single range images from one KITTI sequence (e.g., 00).
    You can wrap this later to create pairs.
    """
    def __init__(self, seq='00', H=64, W=1024, transform=None, lidar_root=None):
        if lidar_root is None:
            lidar_root = KITTI_LIDAR_ROOT
        self.seq_dir = os.path.join(lidar_root, seq, 'velodyne')
        self.files = sorted(glob.glob(os.path.join(self.seq_dir, '*.bin')))
        self.H = H
        self.W = W
        self.transform = transform

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        bin_path = self.files[idx]
        points, intensity = read_velodyne_bin(bin_path)
        range_img = cloud_to_range_image(points, H=self.H, W=self.W)

        # normalize range to [0,1] roughly (clip to max range)
        max_range = 80.0
        range_img = np.clip(range_img, 0, max_range) / max_range

        if self.transform:
            range_img = self.transform(range_img)

        # add channel dimension: (1, H, W)
        range_img = torch.from_numpy(range_img).unsqueeze(0).float()
        return range_img, idx

import random
from torch.utils.data import Dataset

import os
import numpy as np

def load_kitti_positions(seq="00", poses_root=None):
    """
    Load KITTI Odometry poses and return camera positions (N, 3).

    base_dir: path to your data folder (the same one used for KittiRangeDataset),
              expected structure: base_dir/poses/00.txt, etc.
    seq: sequence id as string, e.g. '00'

    Returns:
        positions: np.ndarray of shape (N, 3)
    """
    if poses_root is None:
        poses_root = KITTI_POSES_ROOT
    pose_file = os.path.join(poses_root, "poses", f"{seq}.txt")  # pose file for this sequence
    if not os.path.isfile(pose_file):
        raise FileNotFoundError(f"Pose file not found: {pose_file}")

    poses = []
    with open(pose_file, "r") as f:
        for line in f:
            vals = list(map(float, line.strip().split()))
            if len(vals) != 12:
                continue  # skip malformed lines
            # KITTI stores each pose as 3x4 matrix row-major (R|t)
            T = np.eye(4, dtype=np.float32)
            T[:3, :4] = np.array(vals, dtype=np.float32).reshape(3, 4)
            poses.append(T)

    poses = np.stack(poses, axis=0)              # (N, 4, 4)
    positions = poses[:, :3, 3]                  # (N, 3) translation part
    return positions


import random
import torch
from torch.utils.data import Dataset
# KittiRangeDataset and load_kitti_positions are defined above in this same file


class KittiPairDataset(Dataset):
    """
    Pair dataset for loop-closure style classification.

    Positive: frames within 'pos_range' frames (time-based, simple approximation).
    Negative: frames further than 'neg_gap' frames away in time AND
              farther than 'min_neg_dist' meters in pose space (using KITTI poses).
    """
    def __init__(self, seq='00', H=64, W=1024,
                 pos_range=5, neg_gap=100, num_pairs=5000,
                 min_neg_dist=15.0, use_pose_filter=True, max_neg_trials=20,
                 lidar_root=None, poses_root=None):
        self.single = KittiRangeDataset(seq, H, W, lidar_root=lidar_root)   # underlying single-frame dataset
        self.N = len(self.single)                              # total number of frames in this sequence
        self.pos_range = pos_range                             # how many frames ahead to consider as positive
        self.neg_gap = neg_gap                                 # minimal frame gap for candidate negatives
        self.num_pairs = num_pairs                             # how many pairs this dataset will generate

        self.use_pose_filter = use_pose_filter                 # whether to use pose-based filtering for negatives
        self.min_neg_dist = min_neg_dist                       # minimal spatial distance (meters) for negatives
        self.max_neg_trials = max_neg_trials                   # max resampling attempts for safe negative

        if self.use_pose_filter:
            # Precompute camera positions from KITTI pose file
            self.positions = load_kitti_positions(seq, poses_root=poses_root)   # shape (N, 3)
            # In KITTI Odometry, number of poses may be <= number of velodyne frames.
            # We will clamp indices to valid range later.
            self.num_pose = self.positions.shape[0]
        else:
            self.positions = None
            self.num_pose = 0

    def __len__(self):
        return self.num_pairs                                  # dataset length is number of pairs we want

    def _get_position(self, idx: int):
        """Safely get position for frame idx (clamp if needed)."""
        if self.positions is None:
            return None
        # clamp index in case velodyne frames > pose entries
        idx_clamped = min(idx, self.num_pose - 1)
        return self.positions[idx_clamped]

    def __getitem__(self, _):
        # randomly decide positive or negative pair
        is_pos = random.random() < 0.5

        # randomly pick an anchor frame index
        i = random.randint(0, self.N - 1)

        if is_pos:
            # ----- POSITIVE: j is close to i in time -----
            delta = random.randint(1, self.pos_range)
            j = min(self.N - 1, i + delta)                     # ensure j is in range
            label = 1
        else:
            # ----- NEGATIVE: j is far in time AND far in pose -----
            label = 0

            # candidate indices far in time (same as your original logic)
            choices = list(range(0, max(0, i - self.neg_gap))) + \
                      list(range(min(self.N - 1, i + self.neg_gap), self.N))
            if len(choices) == 0:
                # if sequence is too short / edge case, fallback to fully random
                j = random.randint(0, self.N - 1)
            else:
                j = random.choice(choices)

            if self.use_pose_filter and self.positions is not None:
                # Try to find a j whose pose is at least 'min_neg_dist' away from i
                pos_i = self._get_position(i)
                # In rare cases there might not be any such frames in 'choices', so limit trials
                for _ in range(self.max_neg_trials):
                    pos_j = self._get_position(j)
                    if pos_i is None or pos_j is None:
                        break  # cannot check distance, accept as is

                    dist = float(torch.linalg.norm(torch.tensor(pos_i - pos_j)))
                    if dist >= self.min_neg_dist:
                        # good negative: spatially far enough
                        break
                    # otherwise resample j from 'choices'
                    j = random.choice(choices)

        # fetch the actual range images
        img_i, _ = self.single[i]                              # (1, H, W)
        img_j, _ = self.single[j]                              # (1, H, W)

        # stack them as channels: (2, H, W)
        pair = torch.cat([img_i, img_j], dim=0)

        # return pair and label (float for BCE loss)
        return pair, torch.tensor(label, dtype=torch.float32)
