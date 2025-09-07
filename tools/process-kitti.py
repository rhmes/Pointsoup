import os
import numpy as np
import open3d as o3d

def read_kitti_bin(bin_path):
    """Reads a single KITTI .bin file and returns Nx3 point cloud."""
    scan = np.fromfile(bin_path, dtype=np.float32).reshape(-1, 4)  # (N,4) [x,y,z,intensity]
    points = scan[:, :3]  # Keep only xyz
    return points

def convert_kitti_bin_to_ply(bin_folder, ply_folder):
    os.makedirs(ply_folder, exist_ok=True)

    bin_files = [f for f in os.listdir(bin_folder) if f.endswith('.bin')]
    bin_files.sort()

    for i, bin_file in enumerate(bin_files):
        bin_path = os.path.join(bin_folder, bin_file)
        points = read_kitti_bin(bin_path)

        # Create Open3D point cloud
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)

        # Save as .ply directory if not exists
        if not os.path.exists(ply_folder):
            os.makedirs(ply_folder)
        # Create ply output
        ply_path = os.path.join(ply_folder, f"{os.path.splitext(bin_file)[0]}.ply")
        o3d.io.write_point_cloud(ply_path, pcd)

        print(f"[{i+1}/{len(bin_files)}] Saved {ply_path}")

# Example usage:
bin_folder = "./data/Kitti/2011_09_26/2011_09_26_drive_0018_sync/velodyne_points/data/"
ply_folder = "./data/kitti_ply"
convert_kitti_bin_to_ply(bin_folder, ply_folder)
