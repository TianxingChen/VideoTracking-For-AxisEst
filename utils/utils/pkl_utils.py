import cv2
import torch
import numpy as np
import supervision as sv
from PIL import Image
import open3d as o3d

from utils.submodules.GroundingDINO.utils.video_utils import create_video_from_images

import os
import shutil
import cloudpickle


def load_pkl(
    pkl_path
):
    with open(pkl_path, "rb") as f:
            pkl = cloudpickle.load(f)
    return pkl


class PKL_Utils():
    def __init__(
        self, 
        camera_name_list, 
        skip_hand_camera,       
    ):
        self.camera_name_list = camera_name_list
        self.skip_hand_camera = skip_hand_camera

    def save_obs_as_img(
        self, 
        obs, 
        img_path, 
        img_name
    ):
        if not os.path.exists(img_path):
            os.makedirs(img_path)
        
        for camera_name in self.camera_name_list:
            if self.skip_hand_camera and (camera_name == "hand"):
                continue
            
            camera_path = os.path.join(img_path, camera_name)
            if not os.path.exists(camera_path):
                os.makedirs(camera_path)

            rgb = obs[camera_name]["rgb"]
            if not isinstance(rgb, Image.Image):
                rgb = Image.fromarray(
                    np.uint8(rgb)
                )

            image_path = os.path.join(camera_path, img_name)
            rgb.save(image_path)

    def translate_obs_to_cameras_dict(
        self, 
        obs
    ):
        cameras_dict = {}

        for camera_name in self.camera_name_list:
            if self.skip_hand_camera and (camera_name == "hand"):
                continue

            rgb = obs[camera_name]["rgb"]
            img = Image.fromarray(
                np.uint8(rgb)
            )
            cameras_dict[camera_name] = img

        return cameras_dict

    def translate_tracking_segments(
        self, 
        num_frames, 
        cameras_segment_dic,
        skip_hand_camera = True
    ):
        """
        Func:
            Translate camera_segment_dic (obtained from SAM2) 
                to segment_dicts (used for `get_masked_pcd()`)
        """
        
        segment_dicts_list = []

        for frame_idx in range(num_frames):
            segment_dicts = {}

            # transform masks (bool) to segment (int)
            for camera_name in self.camera_name_list:
                if skip_hand_camera and (camera_name == "hand"):
                    continue

                # len(object_id_list) = num_frames
                object_id_list = cameras_segment_dic[camera_name]["object_id_list"]
                # len(masks_list) = num_frames
                # masks_list[i].shape = [num_objects, h, w] (bool)
                masks_list = cameras_segment_dic[camera_name]["masks_list"]

                target_object_ids = object_id_list[frame_idx]
                target_mask_list = masks_list[frame_idx]

                segment = np.zeros(
                    target_mask_list.shape[1 :], 
                    dtype = np.uint8
                )

                for i, target_object_id in enumerate(target_object_ids):
                    target_mask = target_mask_list[i]
                    segment[target_mask] = target_object_id

                segment_dicts[camera_name] = {
                    "target_object_ids": target_object_ids, 
                    "segment": segment
                }
            
            segment_dicts_list.append(segment_dicts)
        
        return segment_dicts_list

    def pc_camera_to_world(
        self,
        pc, 
        extrinsic
    ):
        R = extrinsic[: 3, : 3]
        T = extrinsic[: 3, 3]
        pc = (R @ pc.T).T + T

        return pc

    def translation_point_cloud(
        self, 
        depth_map, 
        rgb_image, 
        camera_intrinsic, 
        cam2world_matrix, 
        actor_seg
    ):
        depth_map = depth_map.reshape(depth_map.shape[0], depth_map.shape[1])
        rows, cols = depth_map.shape[0], depth_map.shape[1]
        u, v = np.meshgrid(np.arange(cols), np.arange(rows))
        z = depth_map
        x = (u - camera_intrinsic[0][2]) * z / camera_intrinsic[0][0]
        y = (v - camera_intrinsic[1][2]) * z / camera_intrinsic[1][1]
        points = np.dstack((x, y, z))
        per_point_xyz = points.reshape(-1, 3)
        mask = actor_seg
        actor_seg = actor_seg.reshape(-1)
        per_point_rgb = rgb_image.reshape(-1, 3)
        assert (per_point_xyz.shape[0] == actor_seg.shape[0]) \
            and (actor_seg.shape[0] == per_point_rgb.shape[0])
        
        point_xyz = per_point_xyz
        point_rgb = per_point_rgb

        if len(point_xyz) > 0:
            pcd_camera = point_xyz
            seg_mask = np.concatenate(mask)

            pcd_world = self.pc_camera_to_world(pcd_camera, cam2world_matrix)
            return pcd_world, point_rgb, seg_mask
        else:
            return None, None, None

    def get_point_cloud(
        self, 
        obs, 
        segment_dicts, 
    ):
        camera_dicts = {}

        for camera_name in self.camera_name_list:
            if camera_name == "hand" or camera_name == "top":
                continue

            camera_dicts[camera_name] = {}

            camera_dict = obs[camera_name]

            camera_intrinsic = camera_dict["intrinsic"]
            cam2world_matrix = camera_dict["model"]

            Rtilt_rot = cam2world_matrix[: 3, : 3] \
                @ np.array([[1, 0, 0], [0, -1, 0], [0, 0, -1]])
            Rtilt_trl = cam2world_matrix[: 3, 3]
            cam2_wolrd = np.eye(4)
            cam2_wolrd[: 3, : 3] = Rtilt_rot
            cam2_wolrd[: 3, 3] = Rtilt_trl
            camera_dicts[camera_name]["cam2world"] = cam2_wolrd

            camera_rgb = camera_dict["rgb"]
            camera_depth = camera_dict["depth"]
            actor_seg = segment_dicts[camera_name]["segment"]

            point_cloud_world, per_point_rgb, per_seg_mask \
                = self.translation_point_cloud(camera_depth, camera_rgb,
                                            camera_intrinsic, cam2_wolrd,
                                            actor_seg)

            camera_dicts[camera_name]["point_cloud_world"] = point_cloud_world
            camera_dicts[camera_name]["per_point_rgb"] = per_point_rgb
            camera_dicts[camera_name]["rgb"] = camera_rgb
            camera_dicts[camera_name]["depth"] = camera_depth
            camera_dicts[camera_name]["camera_intrinsic"] = camera_intrinsic

            camera_dicts[camera_name]["per_seg_mask"] = per_seg_mask
            camera_dicts[camera_name]["segmentation"] = actor_seg

        return camera_dicts

    def get_scene_point_cloud(
        self, 
        obs, 
        multi_view = False, 
        camera_list = None, 

        # used for tracking the point cloud
        segment_dicts = None,  # segment from SAM2
        target_object_ids_list = None, # the objects you want to track
    ):
        camera_dicts = self.get_point_cloud(
            obs = obs, 
            segment_dicts = segment_dicts
        )
        
        if multi_view and (camera_list is None):
            object_pcd = []

            for camera in camera_dicts:
                pcd_world = camera_dicts[camera]["point_cloud_world"]
                segment = segment_dicts[camera]["segment"]
                segment = segment.reshape(-1)

                for target_object_id in target_object_ids_list:
                    object_pcd.append(pcd_world[segment == target_object_id])

            scene_object_pcd = np.concatenate(object_pcd)
        else:
            raise NotImplementedError(
                "Only support multi_view = True and camera_list = None"
            )
        
        return scene_object_pcd

    def get_masked_pcd(
        self, 
        obs, 
        segment_dicts,  # translated segment
        save_pcd = False, 
        masked_pcds_path = None, 
        pcd_name = None
    ):
        if save_pcd:
            if not os.path.exists(masked_pcds_path):
                os.makedirs(masked_pcds_path)

        masked_pcd = self.get_scene_point_cloud(
            obs, 
            multi_view = True, 
            segment_dicts = segment_dicts, 
            # the object of interest with the highest confidence is indexed 1
            target_object_ids_list = [1]
        )

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(masked_pcd)

        if save_pcd:
            o3d.io.write_point_cloud(
                os.path.join(
                    masked_pcds_path, 
                    f"{os.path.splitext(pcd_name)[0]}.pcd"
                ), 
                pcd
            )
        
        return pcd
