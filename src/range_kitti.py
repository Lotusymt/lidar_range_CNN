import numpy as np
import glob
import os

def read_velodyne_bin(path: str):
    """Read KITTI .bin file -> (N, 4) array of [x, y, z, intensity]."""
    scan = np.fromfile(path, dtype=np.float32).reshape(-1, 4)
    xyz = scan[:, :3]
    intensity = scan[:, 3]
    return xyz, intensity


def cloud_to_range_image(points_xyz: np.ndarray,
                         H: int = 64,
                         W: int = 1024,
                         fov_up_deg: float = 2.0,
                         fov_down_deg: float = -24.8):
    """
    Simple spherical projection to a range image.
    points_xyz: (N, 3) array.
    Returns range_img with shape (H, W), values = range (meters), 0 = empty.
    """
    x, y, z = points_xyz[:, 0], points_xyz[:, 1], points_xyz[:, 2]
    r = np.linalg.norm(points_xyz, axis=1)

    # azimuth (horizontal angle) in [-pi, pi]
    az = np.arctan2(y, x)
    # elevation (vertical angle)
    dist_xy = np.sqrt(x**2 + y**2)
    el = np.arctan2(z, dist_xy)

    fov_up = np.deg2rad(fov_up_deg)
    fov_down = np.deg2rad(fov_down_deg)
    fov = abs(fov_down) + abs(fov_up)

    # Filter points that are within the vertical FOV
    # Only project points that are within the sensor's field of view
    valid_mask = (el >= fov_down) & (el <= fov_up)
    
    if not np.any(valid_mask):
        # No valid points, return empty range image
        return np.zeros((H, W), dtype=np.float32)
    
    # Only process valid points
    az_valid = az[valid_mask]
    el_valid = el[valid_mask]
    r_valid = r[valid_mask]

    # project to image coordinates
    u = 0.5 * (az_valid / np.pi + 1.0)   # should be in [0,1]
    v = 1.0 - ( (el_valid + abs(fov_down)) / fov )  # should be in [0,1], flip so up is small v
    
    # Clamp to [0,1] to handle floating-point precision edge cases
    u = np.clip(u, 0.0, 1.0)
    v = np.clip(v, 0.0, 1.0)

    u_img = (u * (W - 1)).astype(np.int32)
    v_img = (v * (H - 1)).astype(np.int32)

    # initialize range image
    range_img = np.zeros((H, W), dtype=np.float32)

    # keep the closest point per pixel
    for i in range(len(r_valid)):
        col = u_img[i]
        row = v_img[i]
        d = r_valid[i]
        old = range_img[row, col]
        if old == 0 or d < old:
            range_img[row, col] = d

    return range_img
