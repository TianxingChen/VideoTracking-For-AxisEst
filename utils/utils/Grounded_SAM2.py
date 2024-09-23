import os
import shutil

import numpy as np
import torch
import matplotlib.pyplot as plt
from PIL import Image
import supervision as sv
import cv2
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection 

from utils.submodules.SAM2.sam2.build_sam import build_sam2_video_predictor, build_sam2
from utils.submodules.SAM2.sam2.sam2_image_predictor import SAM2ImagePredictor

from utils.submodules.GroundingDINO.utils.video_utils import create_video_from_images


class Grounding_DINO():
    def __init__(
        self, 
        device
    ):
        self.device = device

        # init grounding dino model from huggingface
        model_id = "IDEA-Research/grounding-dino-tiny"
        self.processor = AutoProcessor.from_pretrained(model_id)
        self.grounding_model = AutoModelForZeroShotObjectDetection \
            .from_pretrained(model_id).to(self.device)


class SAM2():
    def __init__(
        self, 
        device
    ):
        import hydra
        hydra.core.global_hydra.GlobalHydra.instance().clear()
        from hydra import initialize_config_module
        initialize_config_module("sam2_configs", version_base="1.2")

        self.device = device

        self.__SAM2_init()

        sam2_checkpoint = "utils/submodules/SAM2/checkpoints/sam2_hiera_large.pt"
        model_cfg = "sam2_hiera_l.yaml"

        self.video_predictor = build_sam2_video_predictor(model_cfg, sam2_checkpoint)
        sam2_image_model = build_sam2(model_cfg, sam2_checkpoint)
        self.image_predictor = SAM2ImagePredictor(sam2_image_model)

        self.OBJECTS = None

    def __SAM2_init(self):
    # Step 1 (Part): Environment settings and model initialization. 

        # use bfloat16 for the entire notebook
        torch.autocast(device_type="cuda", dtype=torch.bfloat16).__enter__()

        if torch.cuda.get_device_properties(0).major >= 8:
            # turn on tfloat32 for Ampere GPUs (https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices)
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True


class Grounded_SAM2():
    def __init__(
        self, 
        device, 
        camera_name_list, 
        skip_hand_camera
    ):
        self.device = device

        self.gdino = Grounding_DINO(self.device)
        self.sam2 = SAM2(self.device)

        self.camera_name_list = camera_name_list
        self.skip_hand_camera = skip_hand_camera

        self.lst = 0
        self.OBJECTS = None
    
    def delete(self):
        del self.gdino
        del self.sam2   

        torch.cuda.empty_cache()
    
    def reset(self):
        self.lst = 0
        self.OBJECTS = None

    def get_tracking_masks_from_cameras_dicts_list_root(
        self, 
        cameras_dicts_list,  # a list of obs_cameras dictionary (sorted in continuous order)
        target_name,  # the object you want to track
        first_frame_idx = 0,  # index of the first frame for SAM2 to start tracking
        save_dir = None, 
        save_JPG = False,  # whether to save visualizations as JPG
        save_MP4 = False,  # whether to save visualizations as MP4
        box_prompt = None,  # bounding box of the handle predicted by RGBManip
        pre_masks = None
    ):
        # build directories for saving visualizations
        if save_JPG or save_MP4:
            if os.path.exists(save_dir):
                shutil.rmtree(save_dir)
            os.makedirs(save_dir)

            if save_JPG:
                for camera_name in self.camera_name_list:
                    if self.skip_hand_camera and (camera_name == "hand"):
                        continue

                    os.makedirs(
                        os.path.join(save_dir, camera_name)
                    )

        # PIL objects of frames from different cameras
        cameras_images_list_dict = {}
        for camera_name in self.camera_name_list:
            cameras_images_list_dict[camera_name] = []

        for cameras_dict in cameras_dicts_list:
            for camera_name in self.camera_name_list:
                if self.skip_hand_camera and (camera_name == "hand"):
                    continue

                rgba = cameras_dict[camera_name]
                img = Image.fromarray(
                    np.uint8(rgba)
                )
                cameras_images_list_dict[camera_name].append(img)

        # IMPORTANT: Text queries need to be lowercased and end with a dot. 
        text = target_name
        if text[-1] != '.':
            text += "."

        cameras_segment_dic = {}

        for camera_name in self.camera_name_list:
            if camera_name == "hand":
                continue

            cameras_segment_dic[camera_name] = {
                "object_id_list": [], 
                "masks_list": []
            }

            cameras_images_list = cameras_images_list_dict[camera_name]
            
            object_id_list, masks_list = self.get_tracking_masks_from_cameras_dicts_list(
                cameras_images_list = cameras_images_list, 
                text = text, 
                first_frame_idx = first_frame_idx, 
                save_dir = save_dir, 
                dir_name = camera_name, 
                save_JPG = save_JPG, save_MP4 = save_MP4, 
                box_prompt = box_prompt, 
                pre_masks = pre_masks
            )

            cameras_segment_dic[camera_name]["object_id_list"] = object_id_list
            cameras_segment_dic[camera_name]["masks_list"] = masks_list
        
        return cameras_segment_dic

    def get_tracking_masks_from_cameras_dicts_list(
        self, 
        cameras_images_list,  # PIL objects of frames from a single camera
        text,  # text queries in SAM2
        first_frame_idx = 0,  # index of the first frame for SAM2 to start tracking
        save_dir = None, 
        dir_name = None, 
        save_JPG = False,  # whether to save visualizations as JPG
        save_MP4 = False,  # whether to save visualizations as MP4
        box_prompt = None,  # bounding box of the handle predicted by RGBManip
        pre_masks = None
    ):
    # Step 1 (Rest): Environment settings and model initialization.

        # Initialize the video_predictor. 
        inference_state = self.sam2.video_predictor.init_state_1(
            cameras_images_list
        )
    # Step 2: Prompt Grounding DINO and SAM image predictor to get the box and mask for specific frame. 

        # the first frame for SAM2 to start tracking
        image = Image.fromarray(
            np.uint8(cameras_images_list[first_frame_idx])
        )
        if image.mode == "RGBA":
            image = image.convert("RGB")

        if (box_prompt is None) and (pre_masks is None):
            # Run Grounding DINO on the first frame. 
            inputs = self.gdino.processor(
                images = image, 
                text = text, 
                return_tensors = "pt"
            ).to(self.device)
            with torch.no_grad():
                outputs = self.gdino.grounding_model(**inputs)

            results = self.gdino.processor.post_process_grounded_object_detection(
                outputs,
                inputs.input_ids,
                box_threshold = 0.25,
                text_threshold = 0.3,
                target_sizes = [image.size[: : -1]]
            )

            # Process the detection results. 
            input_boxes = results[0]["boxes"].cpu().numpy()
            OBJECTS = results[0]["labels"]
            self.OBJECTS = OBJECTS
        
        OBJECTS = self.OBJECTS

        if pre_masks is None:
            # Prompt SAM 2 image predictor to get the mask for the object. 
            # Set image of the SAM2 image predictor. 
            self.sam2.image_predictor.set_image(
                np.array(
                    image.convert("RGB")
                )
            )
            masks, scores, logits = self.sam2.image_predictor.predict(
                point_coords = None,
                point_labels = None,
                box = input_boxes,
                multimask_output = False,
            )
        else:
            masks = pre_masks

        if isinstance(masks, list):
            masks = np.asarray(masks)

        # convert the mask shape to (n, H, W)
        if masks.ndim == 3:
            masks = masks[None]
            scores = scores[None]
            logits = logits[None]
            
        # print(f"? {masks} -> {masks.shape}")

        if masks.ndim == 4:
            # masks = masks.squeeze(1)

            if 1 in masks.shape:
                masks = masks.squeeze(1)
            else:
                masks = masks[:, 0, :, :]

            # if masks.shape[1] > 1:
            #     masks = masks[:, 0, :, :]
            # else:
            #     masks = masks.squeeze(1)

    # Step 3: Register each object's positive points to video predictor with seperate add_new_points call. 
        PROMPT_TYPE_FOR_VIDEO = "mask"

        if PROMPT_TYPE_FOR_VIDEO == "mask":
            if pre_masks is None:
                for object_id, (label, mask) in enumerate(zip(OBJECTS, masks), start = 1):
                    labels = np.ones((1), dtype = np.int32)

                    _, out_obj_ids, out_mask_logits = self.sam2.video_predictor.add_new_mask(
                        inference_state = inference_state,
                        frame_idx = first_frame_idx, # 0
                        obj_id = object_id,
                        mask = mask
                    )
            else:
                object_id = 1
                for i in range(0, min(masks.shape[0], len(cameras_images_list))):
                    mask = masks[i]
                    _, out_obj_ids, out_mask_logits = self.sam2.video_predictor.add_new_mask(
                        inference_state = inference_state,
                        frame_idx = i,
                        obj_id = object_id,
                        mask = mask
                    )
                self.lst = masks.shape[0]
        else:
            raise NotImplementedError()


    # Step 4: Propagate the video predictor to get the segmentation results for each frame. 
        # video_segments contains the per-frame segmentation results
        video_segments = {}

        for out_frame_idx, out_obj_ids, out_mask_logits \
            in self.sam2.video_predictor.propagate_in_video(inference_state):
            video_segments[out_frame_idx] = {
                out_obj_id: (out_mask_logits[i] > 0.0).cpu().numpy()
                for i, out_obj_id in enumerate(out_obj_ids)
            }

    # Step 5: Visualize the segment results across the video and save them. 
        object_id_list = []  # list of object_ids in each frame
        masks_list = []  # mask of each frame

        ID_TO_OBJECTS = {i: obj for i, obj in enumerate(OBJECTS, start = 1)}
        for frame_idx, segments in video_segments.items():
            img_np = np.array(cameras_images_list[frame_idx])
            img = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
            
            object_ids = list(segments.keys())
            masks = list(segments.values())
            masks = np.concatenate(masks, axis = 0)
            
            object_id_list.append(object_ids)
            masks_list.append(masks)

            if save_JPG or save_MP4:
                detections = sv.Detections(
                    xyxy = sv.mask_to_xyxy(masks),  # (n, 4)
                    mask = masks, # (n, h, w)
                    class_id = np.array(object_ids, dtype = np.int32),
                )

                """
                # box
                box_annotator = sv.BoxAnnotator()
                annotated_frame = box_annotator.annotate(
                    scene = img.copy(), 
                    detections = detections
                )

                # label
                label_annotator = sv.LabelAnnotator()
                annotated_frame = label_annotator.annotate(
                    annotated_frame, 
                    detections = detections, 
                    labels = [ID_TO_OBJECTS[i] for i in object_ids]
                )
                """

                # mask
                mask_annotator = sv.MaskAnnotator()
                annotated_frame = mask_annotator.annotate(
                    # scene = annotated_frame, 
                    scene = img.copy(), 
                    detections = detections
                )

                cv2.imwrite(
                    os.path.join(save_dir, dir_name, f"annotated_frame_{frame_idx:05d}.jpg"), 
                    annotated_frame
                )

    # Step 6: Convert the annotated frames to video. 
        if save_MP4:
            output_video_path = os.path.join(save_dir, f"{dir_name}.mp4")
            create_video_from_images(
                os.path.join(save_dir, dir_name), 
                output_video_path
            )
        
        return object_id_list, masks_list
