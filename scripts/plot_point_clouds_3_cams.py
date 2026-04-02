#!/usr/bin/env python3
import argparse
from pathlib import Path
import numpy as np
import open3d as o3d

def load_cloud(path: Path) -> o3d.geometry.PointCloud:
    cloud = o3d.io.read_point_cloud(str(path))
    if cloud.is_empty():
        raise RuntimeError(f"Point cloud is empty or unreadable: {path}")
    return cloud


def describe_cloud(name: str, path: Path, cloud: o3d.geometry.PointCloud) -> None:
    print(f"{name}: {path} ({len(cloud.points)} points)")

def make_y_axis_up(cloud: o3d.geometry.PointCloud) -> o3d.geometry.PointCloud:
    cloud_vis = o3d.geometry.PointCloud(cloud)
    T_vis = np.eye(4, dtype=np.float64)
    T_vis[1, 1] = -1
    T_vis[2, 2] = -1
    cloud_vis.transform(T_vis)
    return cloud_vis


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Visualize pre-ICP and post-ICP point clouds for 3 cameras."
    )
    parser.add_argument("--cloud0"  , default="./data/trial19/debug_calib/pcd_camera_0_wrt_colorcam0.ply"     , help="Pre-ICP cloud 0 path")
    parser.add_argument("--cloud1"  , default="./data/trial19/debug_calib/pcd_camera_1_wrt_colorcam0_before_refinement.ply"     , help="Pre-ICP cloud 1 path")
    parser.add_argument("--cloud2"  , default="./data/trial19/debug_calib/pcd_camera_2_wrt_colorcam0_before_refinement.ply"     , help="Pre-ICP cloud 2 path")
    parser.add_argument("--icp0"    , default="./data/trial19/debug_calib/pcd_camera_0_wrt_colorcam0.ply" , help="Post-ICP cloud 0 path")
    parser.add_argument("--icp1"    , default="./data/trial19/debug_calib/pcd_camera_1_wrt_colorcam0_after_refinement.ply" , help="Post-ICP cloud 1 path")
    parser.add_argument("--icp2"    , default="./data/trial19/debug_calib/pcd_camera_2_wrt_colorcam0_after_refinement.ply" , help="Post-ICP cloud 2 path")
    parser.add_argument(
        "--pre-window-name",
        default="Pre-ICP: cloud_0 + cloud_1 + cloud_2",
        help="Window title for pre-ICP visualization",
    )
    parser.add_argument(
        "--post-window-name",
        default="Post-ICP: icp_cloud_0 + icp_cloud_1 + icp_cloud_2",
        help="Window title for post-ICP visualization",
    )
    parser.add_argument("--width", type=int, default=1400, help="Window width in pixels")
    parser.add_argument(
        "--height", type=int, default=900, help="Window height in pixels"
    )
    parser.add_argument(
        "--no-pre", action="store_true", help="Skip pre-ICP visualization window"
    )
    parser.add_argument(
        "--no-post", action="store_true", help="Skip post-ICP visualization window"
    )
    args = parser.parse_args()

    cloud0_path = Path(args.cloud0)
    cloud1_path = Path(args.cloud1)
    cloud2_path = Path(args.cloud2)
    icp0_path = Path(args.icp0)
    icp1_path = Path(args.icp1)
    icp2_path = Path(args.icp2)

    cloud0 = load_cloud(cloud0_path)
    cloud1 = load_cloud(cloud1_path)
    cloud2 = load_cloud(cloud2_path)
    icp0 = load_cloud(icp0_path)
    icp1 = load_cloud(icp1_path)
    icp2 = load_cloud(icp2_path)

    describe_cloud("cloud0", cloud0_path, cloud0)
    describe_cloud("cloud1", cloud1_path, cloud1)
    describe_cloud("cloud2", cloud2_path, cloud2)
    describe_cloud("icp0", icp0_path, icp0)
    describe_cloud("icp1", icp1_path, icp1)
    describe_cloud("icp2", icp2_path, icp2)

    cloud0_vis = make_y_axis_up(cloud0)
    cloud1_vis = make_y_axis_up(cloud1)
    cloud2_vis = make_y_axis_up(cloud2)
    icp0_vis = make_y_axis_up(icp0)
    icp1_vis = make_y_axis_up(icp1)
    icp2_vis = make_y_axis_up(icp2)

    if not args.no_pre:
        o3d.visualization.draw_geometries(
            [cloud0_vis, cloud1_vis, cloud2_vis],
            window_name=args.pre_window_name,
            width=args.width,
            height=args.height,
        )

    if not args.no_post:
        o3d.visualization.draw_geometries(
            [icp0_vis, icp1_vis, icp2_vis],
            window_name=args.post_window_name,
            width=args.width,
            height=args.height,
        )


if __name__ == "__main__":
    main()
