#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Apply DA3 gravity alignment, convert Y-up world to EgoAllo Z-up world,
and translate the aligned floor to z=0.

Input:
    combined_pcd.ply
    camera_poses.txt
    R_align.npy
    Optional: alignment_transform.npz from 01_estimate_gravity_da3.py

Output:
    egoallo_zup_scene_floor0.ply
    egoallo_zup_poses_floor0.npy

Assumed input after R_align:
    X-Z = ground plane
    +Y  = up
    -Y  = gravity

Output coordinate for EgoAllo-style world:
    X-Y = ground plane
    +Z  = up
    -Z  = gravity

Important:
    This script applies rotation first, then estimates the floor height
    from the aligned Z-up point cloud and translates the floor to z=0.

    The same translation is applied to both:
        1. point cloud points
        2. c2w camera pose translations, Ts[:, :3, 3]

    This keeps the relative position between the camera trajectory and
    the scene point cloud unchanged.

    The input camera poses are assumed to be c2w: T_world_cam.

Rotation order:
    R_align maps DA3 raw world -> Y-up aligned world.
    R_yup_to_zup maps Y-up aligned world -> EgoAllo Z-up world.

    Therefore:
        R_final = R_yup_to_zup @ R_align

Full transform:
    p_out = R_final @ p_raw + t_floor
"""

import argparse
from pathlib import Path

import numpy as np
import open3d as o3d


# ============================================================
# Fixed automatic floor-estimation parameters
# ============================================================

# Use low-z points after Z-up alignment as floor candidates.
LOW_Z_CANDIDATE_PERCENTILE = 35.0

# RANSAC plane parameters. Units follow the input point cloud scale.
RANSAC_DISTANCE_THRESHOLD = 0.03
RANSAC_N = 3
RANSAC_NUM_ITERATIONS = 2000

# abs(normal_z)=1 means perfectly horizontal in Z-up coordinates.
MIN_HORIZONTAL_NORMAL_Z = 0.75

# Fallback if RANSAC is unreliable.
FALLBACK_FLOOR_PERCENTILE = 1.0

# Downsample before RANSAC for speed. This does not affect the saved point cloud.
RANSAC_VOXEL_SIZE = 0.03
MAX_RANSAC_POINTS = 250_000


# ============================================================
# Pose utils
# ============================================================

def quat_to_rot_xyzw(q):
    """
    Quaternion format:
        q = [qx, qy, qz, qw]

    Open3D expects:
        [qw, qx, qy, qz]
    """
    q = np.asarray(q, dtype=np.float64)
    qx, qy, qz, qw = q

    return o3d.geometry.get_rotation_matrix_from_quaternion(
        [qw, qx, qy, qz]
    )


def load_da3_poses_c2w(path):
    """
    Load DA3 camera poses as c2w: T_world_cam.

    Supported row formats:
        timestamp tx ty tz qx qy qz qw
        frame_id  tx ty tz qx qy qz qw
        tx ty tz qx qy qz qw
        flattened 3x4 matrix: 12 values
        timestamp + flattened 3x4 matrix: 13 values
        flattened 4x4 matrix: 16 values
        timestamp + flattened 4x4 matrix: 17 values

    Assumptions:
        - All poses are c2w.
        - Quaternion order is xyzw.
    """
    data = np.loadtxt(path)

    if data.ndim == 1:
        data = data[None, :]

    Ts = []

    for row in data:
        row = np.asarray(row, dtype=np.float64)
        n = row.size

        if n == 16:
            T = row.reshape(4, 4)

        elif n == 17:
            T = row[1:].reshape(4, 4)

        elif n == 12:
            T = np.eye(4, dtype=np.float64)
            T[:3, :4] = row.reshape(3, 4)

        elif n == 13:
            T = np.eye(4, dtype=np.float64)
            T[:3, :4] = row[1:].reshape(3, 4)

        elif n >= 7:
            vals = row[-7:]
            t = vals[:3]
            q = vals[3:7]

            R = quat_to_rot_xyzw(q)

            T = np.eye(4, dtype=np.float64)
            T[:3, :3] = R
            T[:3, 3] = t

        else:
            raise ValueError(f"Unsupported pose row format: {row}")

        Ts.append(T)

    return np.stack(Ts, axis=0)


# ============================================================
# Axis conversion
# ============================================================

def get_R_yup_to_zup():
    """
    Convert a Y-up world to an EgoAllo-style Z-up world.

    Input frame:
        X-Z = ground plane
        +Y  = up

    Output frame:
        X-Y = ground plane
        +Z  = up

    Mapping:
        new_x = old_x
        new_y = -old_z
        new_z = old_y
    """
    return np.array(
        [
            [1.0, 0.0,  0.0],
            [0.0, 0.0, -1.0],
            [0.0, 1.0,  0.0],
        ],
        dtype=np.float64,
    )


# ============================================================
# Debug utils
# ============================================================

def finite_points(P):
    P = np.asarray(P, dtype=np.float64)
    return P[np.all(np.isfinite(P), axis=1)]


def print_xyz_ranges(P, title):
    P = finite_points(P)
    if P.size == 0:
        print(f"\n[{title}] no finite points")
        return

    xyz_min = P.min(axis=0)
    xyz_max = P.max(axis=0)
    xyz_range = xyz_max - xyz_min

    print(f"\n[{title}]")
    print(f"  x min/max/range: {xyz_min[0]: .6f} / {xyz_max[0]: .6f} / {xyz_range[0]: .6f}")
    print(f"  y min/max/range: {xyz_min[1]: .6f} / {xyz_max[1]: .6f} / {xyz_range[1]: .6f}")
    print(f"  z min/max/range: {xyz_min[2]: .6f} / {xyz_max[2]: .6f} / {xyz_range[2]: .6f}")


def print_pose_ranges(Ts, title):
    centers = Ts[:, :3, 3]

    xyz_min = centers.min(axis=0)
    xyz_max = centers.max(axis=0)
    xyz_range = xyz_max - xyz_min

    print(f"\n[{title}] camera centers")
    print(f"  x min/max/range: {xyz_min[0]: .6f} / {xyz_max[0]: .6f} / {xyz_range[0]: .6f}")
    print(f"  y min/max/range: {xyz_min[1]: .6f} / {xyz_max[1]: .6f} / {xyz_range[1]: .6f}")
    print(f"  z min/max/range: {xyz_min[2]: .6f} / {xyz_max[2]: .6f} / {xyz_range[2]: .6f}")


# ============================================================
# Apply alignment
# ============================================================

def apply_rotation_to_point_cloud(pcd, R):
    P = np.asarray(pcd.points, dtype=np.float64)
    P_aligned = (R @ P.T).T

    pcd_out = o3d.geometry.PointCloud()
    pcd_out.points = o3d.utility.Vector3dVector(P_aligned)

    if pcd.has_colors():
        pcd_out.colors = pcd.colors

    if pcd.has_normals():
        N = np.asarray(pcd.normals, dtype=np.float64)
        N_aligned = (R @ N.T).T
        pcd_out.normals = o3d.utility.Vector3dVector(N_aligned)

    return pcd_out


def apply_rotation_to_poses_c2w(Ts, R):
    """
    Input:
        Ts[i] = T_world_cam

    Apply a world-frame rotation:
        R_world_new_world_old = R

    Then:
        R_cam_new = R @ R_cam_old
        t_cam_new = R @ t_cam_old
    """
    Ts_out = Ts.copy()

    for i in range(len(Ts)):
        Ts_out[i, :3, :3] = R @ Ts[i, :3, :3]
        Ts_out[i, :3, 3] = R @ Ts[i, :3, 3]

    return Ts_out


def apply_translation_to_point_cloud(pcd, t):
    """Apply a world-frame translation to point cloud points."""
    P = np.asarray(pcd.points, dtype=np.float64)
    P_translated = P + t.reshape(1, 3)

    pcd_out = o3d.geometry.PointCloud()
    pcd_out.points = o3d.utility.Vector3dVector(P_translated)

    if pcd.has_colors():
        pcd_out.colors = pcd.colors

    if pcd.has_normals():
        pcd_out.normals = pcd.normals

    return pcd_out


def apply_translation_to_poses_c2w(Ts, t):
    """
    Input:
        Ts[i] = T_world_cam, i.e. c2w camera pose.

    Apply the same world-frame translation used for the point cloud:
        camera center t_cam_new = t_cam_old + t

    Rotation is unchanged.
    """
    Ts_out = Ts.copy()
    Ts_out[:, :3, 3] += t.reshape(1, 3)
    return Ts_out


# ============================================================
# Automatic floor estimation
# ============================================================

def point_cloud_from_numpy(P):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(P.astype(np.float64))
    return pcd


def maybe_downsample_for_ransac(P):
    pcd = point_cloud_from_numpy(P)

    if RANSAC_VOXEL_SIZE is not None and RANSAC_VOXEL_SIZE > 0:
        pcd = pcd.voxel_down_sample(voxel_size=RANSAC_VOXEL_SIZE)
        P = np.asarray(pcd.points, dtype=np.float64)

    if len(P) > MAX_RANSAC_POINTS:
        rng = np.random.default_rng(0)
        idx = rng.choice(len(P), size=MAX_RANSAC_POINTS, replace=False)
        P = P[idx]

    return point_cloud_from_numpy(P)


def estimate_floor_z_auto(pcd_zup):
    """
    Estimate floor height after the point cloud is already Z-up.

    Strategy:
        1. Keep lower-z points as floor candidates.
        2. Run RANSAC plane fitting on these candidates.
        3. Accept the plane only if it is roughly horizontal.
        4. Use the median z of inlier points as floor_z.
        5. If plane fitting fails, fall back to a low z percentile.
    """
    P = finite_points(np.asarray(pcd_zup.points, dtype=np.float64))
    if P.size == 0:
        raise RuntimeError("No finite points for automatic floor estimation.")

    fallback_floor_z = float(np.percentile(P[:, 2], FALLBACK_FLOOR_PERCENTILE))
    z_candidate_max = float(np.percentile(P[:, 2], LOW_Z_CANDIDATE_PERCENTILE))
    P_low = P[P[:, 2] <= z_candidate_max]

    if len(P_low) < 100:
        return fallback_floor_z, (
            f"fallback percentile {FALLBACK_FLOOR_PERCENTILE:.1f}: "
            f"too few low-z candidate points ({len(P_low)})"
        )

    try:
        pcd_low = maybe_downsample_for_ransac(P_low)
        if len(pcd_low.points) < 100:
            return fallback_floor_z, (
                f"fallback percentile {FALLBACK_FLOOR_PERCENTILE:.1f}: "
                f"too few downsampled candidate points ({len(pcd_low.points)})"
            )

        plane_model, inliers = pcd_low.segment_plane(
            distance_threshold=RANSAC_DISTANCE_THRESHOLD,
            ransac_n=RANSAC_N,
            num_iterations=RANSAC_NUM_ITERATIONS,
        )

        if len(inliers) < 100:
            return fallback_floor_z, (
                f"fallback percentile {FALLBACK_FLOOR_PERCENTILE:.1f}: "
                f"too few RANSAC plane inliers ({len(inliers)})"
            )

        a, b, c, _ = [float(v) for v in plane_model]
        normal = np.array([a, b, c], dtype=np.float64)
        normal_norm = float(np.linalg.norm(normal))
        if normal_norm < 1e-8:
            return fallback_floor_z, f"fallback percentile {FALLBACK_FLOOR_PERCENTILE:.1f}: invalid plane normal"

        normal = normal / normal_norm
        if abs(normal[2]) < MIN_HORIZONTAL_NORMAL_Z:
            return fallback_floor_z, (
                f"fallback percentile {FALLBACK_FLOOR_PERCENTILE:.1f}: "
                f"RANSAC plane is not horizontal enough, normal={normal.tolist()}"
            )

        P_ransac = np.asarray(pcd_low.points, dtype=np.float64)
        P_inlier = P_ransac[np.asarray(inliers, dtype=np.int64)]
        floor_z = float(np.median(P_inlier[:, 2]))

        return floor_z, (
            "RANSAC horizontal low-plane: "
            f"candidate_percentile={LOW_Z_CANDIDATE_PERCENTILE:.1f}, "
            f"inliers={len(inliers)}, normal={normal.tolist()}"
        )

    except Exception as e:
        return fallback_floor_z, f"fallback percentile {FALLBACK_FLOOR_PERCENTILE:.1f}: RANSAC failed: {repr(e)}"


def estimate_floor_z_from_alignment_meta(meta_path, R_final):
    """
    Prefer the floor plane selected by 01_estimate_gravity_da3.py.

    Supported metadata:
        floor_center_da3:
            A point on the final selected floor plane in raw DA3 world.
            floor_z is computed after applying R_final.

        floor_points_aligned:
            Backward-compatible support for older manual alignment files where
            aligned floor points were already saved in EgoAllo Z-up coordinates.
    """
    meta_path = Path(meta_path).expanduser().resolve()
    if not meta_path.exists():
        raise FileNotFoundError(meta_path)

    with np.load(meta_path, allow_pickle=True) as data:
        keys = set(data.keys())

        if "floor_center_da3" in keys:
            floor_center_da3 = np.asarray(data["floor_center_da3"], dtype=np.float64).reshape(3)
            floor_center_zup = R_final @ floor_center_da3
            floor_z = float(floor_center_zup[2])
            return floor_z, f"alignment metadata floor_center_da3 from {meta_path}"

        if "floor_points_aligned" in keys:
            floor_points = np.asarray(data["floor_points_aligned"], dtype=np.float64)
            if floor_points.ndim != 2 or floor_points.shape[1] != 3:
                raise ValueError(
                    f"floor_points_aligned should be [N,3], got {floor_points.shape}"
                )
            floor_points = floor_points[np.isfinite(floor_points).all(axis=1)]
            if len(floor_points) == 0:
                raise ValueError("No finite floor_points_aligned in alignment metadata.")
            floor_z = float(np.median(floor_points[:, 2]))
            return floor_z, f"alignment metadata floor_points_aligned from {meta_path}"

    raise KeyError(
        f"No supported floor metadata found in {meta_path}. "
        "Expected floor_center_da3 or floor_points_aligned."
    )


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description=(
            "Apply DA3 gravity rotation to point cloud and c2w camera poses, "
            "convert Y-up world to EgoAllo Z-up world, and move floor to z=0."
        )
    )

    parser.add_argument("--pcd", required=True, help="Input DA3 point cloud, e.g. combined_pcd.ply")
    parser.add_argument("--poses", required=True, help="Input DA3 c2w camera poses, e.g. camera_poses.txt")
    parser.add_argument("--R", required=True, help="Input Y-up gravity rotation matrix, e.g. R_align.npy")
    parser.add_argument(
        "--align_meta",
        default=None,
        help="Optional alignment metadata .npz from 01_estimate_gravity_da3.py",
    )

    parser.add_argument("--out_pcd", default="egoallo_zup_scene_floor0.ply", help="Output EgoAllo Z-up floor-z0 point cloud")
    parser.add_argument("--out_poses", default="egoallo_zup_poses_floor0.npy", help="Output EgoAllo Z-up floor-z0 c2w poses")
    parser.add_argument("--out_R_final", default=None, help="Optional path to save final rotation matrix")
    parser.add_argument(
        "--output_voxel_size",
        type=float,
        default=0.01,
        help="Voxel size for final point-cloud downsampling; use 0 to disable.",
    )

    args = parser.parse_args()

    # Load point cloud.
    pcd = o3d.io.read_point_cloud(args.pcd)
    if pcd.is_empty():
        raise RuntimeError(f"Empty point cloud: {args.pcd}")

    # Load c2w poses.
    Ts = load_da3_poses_c2w(args.poses)

    # Load R_align: raw DA3 world -> Y-up aligned world.
    R_align = np.load(args.R).astype(np.float64)
    if R_align.shape != (3, 3):
        raise ValueError(f"R must be 3x3, got {R_align.shape}")

    # Convert Y-up aligned world -> EgoAllo Z-up world.
    R_yup_to_zup = get_R_yup_to_zup()

    # Final transform: raw DA3 world -> EgoAllo Z-up world.
    R_final = R_yup_to_zup @ R_align

    det_R = np.linalg.det(R_final)
    if not np.isclose(det_R, 1.0, atol=1e-3):
        print(f"[Warning] det(R_final) = {det_R:.6f}, expected close to 1.0")

    # Debug before alignment.
    P_raw = np.asarray(pcd.points, dtype=np.float64)
    print_xyz_ranges(P_raw, "raw point cloud")
    print_pose_ranges(Ts, "raw poses")

    # Step 1: apply final rotation.
    pcd_rot = apply_rotation_to_point_cloud(pcd, R_final)
    Ts_rot = apply_rotation_to_poses_c2w(Ts, R_final)

    # Debug after rotation, before floor translation.
    P_rot = np.asarray(pcd_rot.points, dtype=np.float64)
    print_xyz_ranges(P_rot, "EgoAllo Z-up point cloud before floor translation")
    print_pose_ranges(Ts_rot, "EgoAllo Z-up poses before floor translation")

    # Step 2: estimate current floor height and translate it to z=0.
    if args.align_meta is not None:
        floor_z, floor_method = estimate_floor_z_from_alignment_meta(
            args.align_meta,
            R_final,
        )
    else:
        floor_z, floor_method = estimate_floor_z_auto(pcd_rot)

    t_floor = np.array([0.0, 0.0, -floor_z], dtype=np.float64)

    print("\n[automatic floor estimation]")
    print(f"  method: {floor_method}")
    print(f"  estimated floor_z before translation: {floor_z:.6f}")
    print(f"  apply translation to both point cloud and camera centers: {t_floor.tolist()}")

    pcd_out = apply_translation_to_point_cloud(pcd_rot, t_floor)
    Ts_out = apply_translation_to_poses_c2w(Ts_rot, t_floor)

    # Debug after alignment and floor translation.
    P_aligned = np.asarray(pcd_out.points, dtype=np.float64)
    print_xyz_ranges(P_aligned, "EgoAllo Z-up point cloud after floor translation")
    print_pose_ranges(Ts_out, "EgoAllo Z-up poses after floor translation")

    if args.output_voxel_size < 0:
        raise ValueError("--output_voxel_size must be >= 0")
    if args.output_voxel_size > 0:
        points_before = len(pcd_out.points)
        pcd_out = pcd_out.voxel_down_sample(args.output_voxel_size)
        print(
            "Downsampled output point cloud: "
            f"{points_before} -> {len(pcd_out.points)} points "
            f"(voxel_size={args.output_voxel_size})"
        )

    # Save.
    o3d.io.write_point_cloud(args.out_pcd, pcd_out)
    np.save(args.out_poses, Ts_out)

    if args.out_R_final is not None:
        np.save(args.out_R_final, R_final)
        print("Saved final rotation:", args.out_R_final)

    print("\nSaved point cloud:", args.out_pcd)
    print("Saved poses:", args.out_poses)
    print("\nDone.")
    print("This script applied:")
    print("  1. R_align: raw DA3 world -> Y-up aligned world")
    print("  2. R_yup_to_zup: Y-up aligned world -> EgoAllo Z-up world")
    print("  3. automatic t_floor: translated estimated floor to z=0")
    print("\nAfter alignment, expected EgoAllo-style convention:")
    print("  X-Y = ground plane")
    print("  +Z = up")
    print("  -Z = gravity")


if __name__ == "__main__":
    main()
