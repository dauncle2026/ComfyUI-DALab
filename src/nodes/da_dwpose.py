'''
@author: 
@date: 2026-01-25
@description: This node is used to detect human poses using DWPose.

required:
- cv2
- matplotlib
'''
import os
import torch
import numpy as np
from tqdm import tqdm
from fractions import Fraction

import folder_paths
import comfy.utils
import comfy.model_management as model_management
from comfy_api.latest import io, InputImpl, Types, Input

from ..utils import utils
from ..utils.config_loader import ConfigLoader
from ..utils.logger import logger
from ..utils.paths import get_config_file_path

_CONFIG_FILE_PATH = get_config_file_path("dwpose")
_GLOBAL_CONFIG_PATH = get_config_file_path("global")

class DADWPoseConfig(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        config = ConfigLoader(_CONFIG_FILE_PATH, strict=False)
        
        detect_path = os.path.join(folder_paths.models_dir, "dalab","dwpose", "yolox_l.torchscript.pt")
        pose_path = os.path.join(folder_paths.models_dir, "dalab", "dwpose", "dw-ll_ucoco_384_bs5.torchscript.pt")

        return io.Schema(
            node_id="DADWPoseConfig",
            display_name="DA DWPose Config",
            category="DALab/Tools/DWPose",
            description="Configure DWPose model settings. Run first to save config.",
            is_output_node=True,
            inputs=[
                io.String.Input(
                    "bbox_detector",
                    default=config.get("bbox_detector", detect_path),
                    display_name="Bbox Detector",
                    tooltip="YOLO model for person detection. TorchScript is faster on GPU.",
                ),
                io.String.Input(
                    "pose_estimator",
                    default=config.get("pose_estimator", pose_path),
                    display_name="Pose Estimator",
                    tooltip="DWPose model for keypoint estimation. TorchScript is faster on GPU.",
                ),
                io.Int.Input(
                    "resolution",
                    default=config.get("resolution", 512),
                    min=64,
                    max=2048,
                    step=64,
                    display_name="Resolution",
                    tooltip="Output resolution for pose detection.",
                ),
            ],
            outputs=[],
        )

    @classmethod
    def execute(
        cls,
        bbox_detector: str,
        pose_estimator: str,
        resolution: int,
    ) -> io.NodeOutput:
        config_data = {
            "bbox_detector": bbox_detector,
            "pose_estimator": pose_estimator,
            "resolution": resolution,
        }
        utils.save_json(config_data, _CONFIG_FILE_PATH)
        return io.NodeOutput()


class DADWPose(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="DADWPose",
            display_name="DA DWPose",
            category="DALab/Tools/DWPose",
            description="Detect human poses using DWPose and generate pose images or videos.",
            is_input_list=True,
            inputs=[
                io.MultiType.Input(
                    "frames",
                    types=[io.Image, io.Video],
                    tooltip="Input images or video for pose detection.",
                ),
                io.Boolean.Input(
                    "detect_body",
                    default=True,
                    display_name="Detect Body",
                    tooltip="Draw body keypoints.",
                ),
                io.Boolean.Input(
                    "detect_hand",
                    default=True,
                    display_name="Detect Hand",
                    tooltip="Draw hand keypoints.",
                ),
                io.Boolean.Input(
                    "detect_face",
                    default=True,
                    display_name="Detect Face",
                    tooltip="Draw face keypoints.",
                ),
            ],
            outputs=[
                io.Image.Output(
                    "images",
                    tooltip="Generated pose visualization image.",
                    is_output_list=True,
                ),
                io.Video.Output(
                    "videos",
                    is_output_list=True,
                    tooltip="Generated pose visualization video.",
                ),
                io.Custom("DWPOSE_KEYPOINTS").Output(
                    "keypoints",
                    tooltip="The keypoints of the detected poses.",
                    is_output_list=True,
                ),
            ],
        )

    @classmethod
    def execute(
        cls,
        frames: list[io.Image.Type] | io.Video.Type,
        detect_body: list[bool],
        detect_hand: list[bool],
        detect_face: list[bool],
    ) -> io.NodeOutput:
        config = ConfigLoader(_CONFIG_FILE_PATH, strict=True)
        global_config = ConfigLoader(_GLOBAL_CONFIG_PATH, strict=False)
        release_after_run = global_config.get("release_after_run", False)

        detect_body = detect_body[0]
        detect_hand = detect_hand[0]
        detect_face = detect_face[0]

        bbox_detector = config.get("bbox_detector")
        pose_estimator = config.get("pose_estimator")
        resolution = config.get("resolution")

        if not os.path.exists(bbox_detector):
            raise FileNotFoundError(f"Bbox detector model not found. Please download model from https://huggingface.co/hr16/yolox-onnx/resolve/main/yolox_l.torchscript.pt to 'models/dalab/dwpose/yolox_l.torchscript.pt'")
        if not os.path.exists(pose_estimator):
            raise FileNotFoundError(f"Pose estimator model not found. Please download model from https://huggingface.co/hr16/DWPose-TorchScript-BatchSize5/resolve/main/dw-ll_ucoco_384_bs5.torchscript.pt to 'models/dalab/dwpose/dw-ll_ucoco_384_bs5.torchscript.pt'")

        try:
            from ..libs.dwpose import DwposeDetector
        except ImportError as e:
            raise ImportError(f"[DALab] Node DADWPose Error: {e}")

        device = model_management.get_torch_device()
        detector = DwposeDetector.from_local(
            det_model_path=bbox_detector,
            pose_model_path=pose_estimator,
            torchscript_device=device,
        )

        is_video = isinstance(frames[0], Input.Video)

        output_videos = []
        output_images = []
        output_keypoints = []

        try:
            for idx, frame in enumerate(frames):
                logger.info(f"DWPose Processing frame: {idx + 1}/{len(frames)}")

                if is_video:
                    components = frame.get_components()
                    images = components.images
                    frame_rate = components.frame_rate
                    audio = components.audio
                else:
                    images = frame
                    frame_rate = None
                    audio = None

                batch_size = images.shape[0]
                pbar = comfy.utils.ProgressBar(batch_size)

                pose_images = []
                keypoints = []

                for i in tqdm(range(batch_size), desc="Processing pose frames", total=batch_size):
                    np_image = (images[i].cpu().numpy() * 255).astype(np.uint8)

                    pose_img, keypoints_dict = detector(
                        np_image,
                        detect_resolution=resolution,
                        include_body=detect_body,
                        include_hand=detect_hand,
                        include_face=detect_face,
                        output_type="np",
                        image_and_json=True,
                    )

                    pose_tensor = torch.from_numpy(pose_img.astype(np.float32) / 255.0)
                    pose_images.append(pose_tensor)
                    keypoints.append(keypoints_dict)

                    pbar.update(1)

                output_frames = torch.stack(pose_images, dim=0)
                output_keypoints.append(keypoints)

                if is_video:
                    video = InputImpl.VideoFromComponents(
                        Types.VideoComponents(
                            images=output_frames,
                            audio=audio,
                            frame_rate=frame_rate if frame_rate else Fraction(24)
                        )
                    )
                    output_videos.append(video)
                else:
                    output_images.append(output_frames)

            return io.NodeOutput(output_images, output_videos, output_keypoints)

        finally:
            if release_after_run:
                DwposeDetector.release()
                logger.info("DWPose models released")

    @classmethod
    def fingerprint_inputs(cls, **kwargs):
        try:
            config_mtime = os.path.getmtime(_CONFIG_FILE_PATH)
            global_config_mtime = os.path.getmtime(_GLOBAL_CONFIG_PATH)
        except OSError:
            config_mtime = 0
            global_config_mtime = 0
            
        return hash((str(kwargs), str(config_mtime), str(global_config_mtime)))
