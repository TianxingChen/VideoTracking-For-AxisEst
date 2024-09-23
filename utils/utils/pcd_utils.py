import torch
import cv2
import numpy as np
import supervision as sv
from PIL import Image
import open3d as o3d

import os
import shutil

import cloudpickle


def normalize(vector):
    length = np.linalg.norm(vector)
    return vector / length if (length > 1e-8) else vector

def get_pcd_num_points(pcd):
    return np.asarray(pcd.points).shape[0]

def save_pcd(
    pcd, 
    save_pcd_path, 
    pcd_name
):
    if isinstance(pcd, np.ndarray):
        point_cloud = o3d.geometry.PointCloud()
        point_cloud.points = o3d.utility.Vector3dVector(pcd)

        pcd = point_cloud
    
    o3d.io.write_point_cloud(
        os.path.join(save_pcd_path, pcd_name), 
        pcd
    )

def get_clean_pcd(
    pcd, 
    nb_points = 200, 
    radius = 0.05
):
    pcd_clean, _ = pcd.remove_radius_outlier(
        nb_points = nb_points, 
        radius = radius
    )

    return pcd_clean

def pcd_minus_obb(
    pcd, 
    obb
):
    obb_center = np.asarray(obb.center)
    obb_half_extent = np.asarray(obb.extent) / 2
    obb_R = np.asarray(obb.R)

    points = np.asarray(pcd.points)
    
    points_local = points - obb_center
    points_local = points_local @ np.linalg.inv(obb_R).T

    mask = np.all(
        np.abs(points_local) <= obb_half_extent, 
        axis = 1
    )
    rest_points = points[~mask]

    pcd.points = o3d.utility.Vector3dVector(rest_points)
    
    return pcd

def get_minimum_bounding_box(pcd):
    def rotate_pcd(
        pcd, 
        R,  # rotation matrix
    ):
        rotated_pcd = o3d.geometry.PointCloud()
        rotated_pcd.points = pcd.points

        rotated_pcd.rotate(
            R, 
            center = np.asarray([0, 0, 0])
        )

        return rotated_pcd
    
    pcd_center = pcd.get_center()
    pcd.translate(-pcd_center)

    minimum_aabb = None
    min_volume = float("inf")
    best_R_z = None

    for angle_z in np.linspace(0, 2 * np.pi, 120):
        R_z = np.asarray(
            [
                [np.cos(angle_z), -np.sin(angle_z), 0], 
                [np.sin(angle_z), np.cos(angle_z), 0], 
                [0, 0, 1]
            ]
        )
        rotated_pcd = rotate_pcd(
            pcd, 
            R = R_z, 
        )

        rotated_aabb = rotated_pcd.get_axis_aligned_bounding_box()
        volume = rotated_aabb.volume()

        if volume < min_volume:
            min_volume = volume
            minimum_aabb = rotated_aabb
            best_R_z = R_z
    
    minimum_obb = minimum_aabb.get_oriented_bounding_box()

    minimum_obb.rotate(
        np.linalg.inv(best_R_z), 
        center = np.asarray([0, 0, 0])
    )

    minimum_obb.translate(pcd_center)
    pcd.translate(pcd_center)

    return minimum_obb

def get_minimum_cuboid_vertices(minimum_obb):
    center = np.asarray(minimum_obb.center)
    half_extent = np.asarray(minimum_obb.extent) / 2
    R = np.asarray(minimum_obb.R)

    local_axes = [
        normalize(R[:, i]) for i in range(3)
    ]

    vertices = []

    bias_list = [
        [-1, -1, -1], 
        [-1, 1, -1], 
        [1, 1, -1], 
        [1, -1, -1], 
        [1, -1, 1], 
        [1, 1, 1], 
        [-1, 1, 1], 
        [-1, -1, 1], 
    ]

    for bias in bias_list:
        i, j, k = bias

        vertex = center \
            + i * local_axes[0] * half_extent[0] + j * local_axes[1] * half_extent[1] + k * local_axes[2] * half_extent[2]
        vertices.append(vertex)

    return np.asarray(vertices)

def combine_pcd_with_minimum_cuboid(
    pcd, 
    vertices, 
    reserve_pcd = True, 
    num_edge_points = 30
):
    def get_segment_points(
        st_point, ed_point, 
        num_points
    ):
        assert num_points >= 2

        arrow = ed_point - st_point
        length = np.linalg.norm(arrow)
        direction = normalize(arrow)

        segment_points = []

        for i in np.linspace(0, length, num_points):
            point = st_point + i * direction
            segment_points.append(point)
        
        return np.asarray(segment_points)

    points = np.asarray([])
    if reserve_pcd:
        points = np.asarray(pcd.points)
    
    edge_pairs = [
        [0, 1], [1, 2], [2, 3], [3, 0], 
        [0, 7], [1, 6], [2, 5], [3, 4], 
        [7, 6], [6, 5], [5, 4], [4, 7]
    ]

    for [i, j] in edge_pairs:
        segment_points = get_segment_points(
            st_point = vertices[i], 
            ed_point = vertices[j], 
            num_points = num_edge_points
        )

        if reserve_pcd:
            points = np.vstack(
                [points, segment_points]
            )
        else:
            points = segment_points

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)

    return pcd

def get_obb_midperpendicular(obb):
    obb_center = np.asarray(obb.center)
    obb_half_extent = np.asarray(obb.extent) / 2

    obb_R = np.asarray(obb.R)
    local_axis_list = [
        normalize(obb_R[:, i]) for i in range(3)
    ]

    vertices = []

    bias_list = [
        [-1, -1, -1], 
        [-1, 1, -1], 
        [1, 1, -1], 
        [1, -1, -1]
    ]

    for bias in bias_list:
        i, j, k = bias

        vertex = obb_center \
            + i * local_axis_list[0] * obb_half_extent[0] + j * local_axis_list[1] * obb_half_extent[1] \
                + k * local_axis_list[2] * obb_half_extent[2]
        vertices.append(vertex)

    longest_axis_idx = np.argmax(obb_half_extent[: -1])
    
    dir = local_axis_list[longest_axis_idx]
    mid = (vertices[0 if (longest_axis_idx == 0) else 2] + vertices[1]) / 2

    dir = dir[: -1]
    mid = mid[: -1]

    return mid, dir

def line_intersection(
    line1, line2  # 2D lines
):
    mid1, dir1 = line1
    mid2, dir2 = line2

    # parallel
    if np.isclose(
        np.cross(dir1, dir2), 
        0
    ):
        return None

    A = np.asarray(
        [
            dir1, 
            -dir2
        ]
    ).T
    b = mid2 - mid1

    # solve equation: A * t = b
    t = np.linalg.solve(A, b)

    return mid1 + t[0] * dir1

def get_unique_point_list(
    point_list, 
    tol = 1e-5
):
    unique_point_list = []

    for point in point_list:
        if not isinstance(point, np.ndarray):
            point = np.asarray(point)

        duplicate = False

        for u_point in unique_point_list:
            if np.isclose(
                np.linalg.norm(point - u_point), 
                0, 
                atol = tol
            ):
                duplicate = True
                break
        
        if not duplicate:
            unique_point_list.append(point)

    return unique_point_list

def get_axis_direction_obb(
    obb_list, 
    estimated_pivot_point
):
    z = np.array(
        [0, 0, 1.0], 
        dtype = np.float32
    )
    
    center_list = [
        obb.center for obb in obb_list
    ]

    src_center = center_list[0]
    dst_center = center_list[-1]
    direction = normalize(dst_center - src_center)

    arrow = estimated_pivot_point - src_center
    tangent_direction = np.cross(arrow, z)

    return z if (np.dot(tangent_direction, direction) > 0) else -z

# may not work
def filter_pcd_handle(
    pcd, 
    hist_threshold = 0.3
):
    def project_to_axis(
        pcd_points_local, 
        axis_idx, 
        num_bins = 100
    ):
        projection = pcd_points_local[:, axis_idx]

        return np.histogram(
            projection, 
            bins = num_bins
        )

    def find_handle_region(
        hist, bin_edges, 
        hist_threshold
    ):
        handle_region_mask = (
            hist < (np.max(hist) * hist_threshold)
        )

        handle_region = bin_edges[: -1][handle_region_mask]

        # no handle region
        if len(handle_region) == 0:
            handle_region = [
                bin_edges[0], 
                bin_edges[1]
            ]

        return handle_region

    def filter_handle(
        pcd_points_local, 
        handle_region_list
    ):
        mask_list = [
            (
                pcd_points_local[:, i] >= np.min(handle_region_list[i])
            ) & \
            (
                pcd_points_local[:, i] <= np.max(handle_region_list[i])
            ) for i in range(3)
        ]

        handle_mask = mask_list[0] & mask_list[1] & mask_list[2]

        return pcd_points_local[~handle_mask]

    minimum_obb = get_minimum_bounding_box(pcd)

    obb_center = np.asarray(minimum_obb.center)
    obb_R = np.asarray(minimum_obb.R)

    pcd_points = np.asarray(pcd.points)

    pcd_points_local = np.dot(
        pcd_points - obb_center, 
        obb_R.T
    )

    hist_list = []
    bin_edges_list = []

    for i in range(3):
        hist, bin_edges = project_to_axis(
            pcd_points_local, 
            axis_idx = i
        )

        hist_list.append(hist)
        bin_edges_list.append(bin_edges)

    handle_region_list = [
        find_handle_region(
            hist_list[i], bin_edges_list[i], 
            hist_threshold = hist_threshold
        ) for i in range(3)
    ]

    filtered_points_local = filter_handle(
        pcd_points_local, 
        handle_region_list
    )

    filtered_points_world = \
        np.dot(filtered_points_local, obb_R) + obb_center

    filtered_pcd = o3d.geometry.PointCloud()
    filtered_pcd.points = o3d.utility.Vector3dVector(filtered_points_world)

    return filtered_pcd

def predict_axis(
    joint_type, 
    rest_pcd_list, 
    minimum_obb_list, 

    valid_num_points_threshold = 500, 

    last_st_idx = None, 
    last_frame_interval = None, 

    obs_num_variety = 1
):
    def check_obb_ratio(
        obb, 
        ratio_threshold = 1.2
    ):
        obb_center = np.asarray(obb.center)
        obb_half_extent = np.asarray(obb.extent) / 2

        obb_R = np.asarray(obb.R)
        local_axis_list = [
            normalize(obb_R[:, i]) for i in range(3)
        ]

        vertices = []

        bias_list = [
            [-1, -1, -1], 
            [-1, 1, -1], 
            [1, 1, -1], 
            [1, -1, -1]
        ]

        for bias in bias_list:
            i, j, k = bias

            vertex = obb_center \
                + i * local_axis_list[0] * obb_half_extent[0] + j * local_axis_list[1] * obb_half_extent[1] \
                    + k * local_axis_list[2] * obb_half_extent[2]
            vertices.append(vertex)

        longest_axis_idx = np.argmax(obb_half_extent[: -1])
        shortest_axis_idx = 1 if (longest_axis_idx == 0) else 0

        if np.isclose(obb_half_extent[shortest_axis_idx], 0, atol = 1e-5):
            return False
        
        return obb_half_extent[longest_axis_idx] >= obb_half_extent[shortest_axis_idx] * ratio_threshold

    print(f"joint_type: {joint_type}")

    ed_idx = len(rest_pcd_list) - 1

    # prismatic joint
    if joint_type == "prismatic":
        if last_st_idx is None:
            st_idx = ed_idx - 1
            while (st_idx - 1 >= 1) and (get_pcd_num_points(rest_pcd_list[st_idx - 1]) >= valid_num_points_threshold):
                st_idx -= 1
        else:
            st_idx = last_st_idx
        
        max_num_frames = 10 * obs_num_variety
        if ed_idx - st_idx + 1 >= max_num_frames:
            st_idx = ed_idx - max_num_frames + 1

        last_st_idx = st_idx
        print(f"valid frame idx range: [{st_idx}, {ed_idx}]")

        src_center = minimum_obb_list[st_idx].center
        dst_center = minimum_obb_list[ed_idx].center

        direction = dst_center - src_center

        direction[2] = 0  # project to xy plane
        direction = normalize(direction)

        estimated_axis_direction = direction if (st_idx != ed_idx) \
            else np.asarray([0, 0, 1], dtype = np.float32)
        estimated_pivot_point = src_center
    # revolute joint
    elif joint_type == "revolute":
        if last_frame_interval is None:
            st_idx = ed_idx - 1
            while (st_idx - 1 >= 1) and check_obb_ratio(minimum_obb_list[st_idx - 1]):
                st_idx -= 1
        else:
            st_idx = max(ed_idx - last_frame_interval + 1, 1)

        max_num_frames = 8 * obs_num_variety
        if ed_idx - st_idx + 1 >= max_num_frames:
            st_idx = ed_idx - max_num_frames + 1

        last_frame_interval = ed_idx - st_idx + 1
        print(f"valid frame idx range: [{st_idx}, {ed_idx}]")

        obb_midperpendicular_list = [
            None for i in range(st_idx)
        ]
        obb_midperpendicular_list += [
            get_obb_midperpendicular(minimum_obb_list[i]) \
                for i in range(st_idx, len(minimum_obb_list))
        ]

        candidate_center_list = []
    
        pre_idx = st_idx

        obb_midperpendicular_1 = obb_midperpendicular_list[pre_idx]
        obb_midperpendicular_2 = obb_midperpendicular_list[-1]

        mid1, dir1 = obb_midperpendicular_1[0], obb_midperpendicular_1[1]
        mid2, dir2 = obb_midperpendicular_2[0], obb_midperpendicular_2[1]
        
        intersecton_point = line_intersection(
            [mid1, dir1], 
            [mid2, dir2]
        )

        if intersecton_point is not None:
            candidate_center_list.append(intersecton_point)

        candidate_center_list = get_unique_point_list(
            candidate_center_list, 
            tol = 1e-5
        )

        if len(candidate_center_list) == 0:
            print("[Failed] No candidate center.")

            candidate_center_list.append(
                np.asarray(
                    [0, 0], 
                    dtype = np.float32
                )
            )

        estimated_pivot_point = candidate_center_list[0]

        estimated_pivot_point = np.hstack(
            [estimated_pivot_point, [0]]
        )

        estimated_axis_direction = get_axis_direction_obb(
            obb_list = minimum_obb_list[st_idx : (ed_idx + 1)], 
            estimated_pivot_point = estimated_pivot_point
        )
    else:
        raise ValueError(f"Joint type error, got {joint_type}.")
    
    return estimated_axis_direction, estimated_pivot_point, \
        last_st_idx, last_frame_interval


