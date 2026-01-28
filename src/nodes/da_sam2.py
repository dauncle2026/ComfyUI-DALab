import os
import json
import numpy as np
import torch
from PIL import Image

import folder_paths
from comfy_api.latest import io, Input

from ..utils import utils
from ..utils.config_loader import ConfigLoader
from ..utils.wrappers.sam2 import get_sam2_image, get_sam2_video
from ..utils.wrappers.base import handle_wrapper_after_run
from ..utils.logger import logger
from ..utils.paths import get_config_file_path

_CONFIG_FILE_PATH = get_config_file_path("sam2")

class DASAM2Config(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        config = ConfigLoader(_CONFIG_FILE_PATH, strict=False)
        model_path = os.path.join(folder_paths.models_dir, "dalab", "sam2", "sam2.1-hiera-large")

        return io.Schema(
            node_id="DASAM2Config",
            display_name="DA SAM2 Config",
            category="DALab/Tools/SAM2",
            description="Configure SAM2 model settings. Run first to save config.",
            is_output_node=True,
            inputs=[
                io.String.Input(
                    "model_path",
                    default=config.get("model_path", model_path),
                    display_name="Model Path",
                    tooltip="Path to SAM2 model directory.",
                ),
                io.Combo.Input(
                    "precision",
                    default=config.get("precision", "bf16"),
                    options=["fp16", "bf16", "fp32"],
                    display_name="Precision",
                    tooltip="Model precision. bf16 recommended for SAM2.",
                ),
            ],
            outputs=[],
        )

    @classmethod
    def execute(
        cls,
        model_path: str,
        precision: str,
    ) -> io.NodeOutput:
        config_data = {
            "model_path": model_path,
            "precision": precision,
        }
        utils.save_json(config_data, _CONFIG_FILE_PATH)
        return io.NodeOutput()


class DASAM2(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="DASAM2",
            display_name="DA SAM2",
            category="DALab/Tools/SAM2",
            description="Segment objects using SAM2 with bbox input from Florence2.",
            is_input_list=True,
            inputs=[
                io.MultiType.Input(
                    "images",
                    types=[io.Image, io.Video],
                    tooltip="Input images or video for SAM2 segmentation.",
                    display_name="images",
                ),
                io.Custom("BBOX_DATA").Input(
                    "bbox_data",
                    display_name="bbox_data",
                    tooltip="Bounding box data from Florence2 (JSON format).",
                ),
                io.Int.Input(
                    "seed",
                    default=0,
                    min=0,
                    max=0xFFFFFFFFFFFFFFFF,
                    display_name="seed",
                    tooltip="Random seed.",
                    control_after_generate=True,
                ),
            ],
            outputs=[
                io.Mask.Output(
                    "masks",
                    tooltip="Segmentation masks.",
                    is_output_list=True,
                ),
            ],
        )

    @classmethod
    def execute(
        cls,
        images: list,
        bbox_data: list[str],
        seed: list[int],
    ) -> io.NodeOutput:
        config = ConfigLoader(_CONFIG_FILE_PATH, strict=True)

        batch_inputs = utils.inputs_to_batch(
            image=images,
            bbox_data=bbox_data,
        )

        model_path = config.get("model_path")
        precision = config.get("precision")

        if not os.path.exists(model_path):
            raise FileNotFoundError(f"SAM2 model not found.Please download model from https://huggingface.co/facebook/sam2-hiera-large to 'models/dalab/sam2/sam2.1-hiera-large'")
        
        config_path = os.path.join(model_path, "config.json")
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"SAM2 model not found.Please download model from https://huggingface.co/facebook/sam2-hiera-large to 'models/dalab/sam2/sam2.1-hiera-large'")
        
        if len(images) == 0:
            raise ValueError("No images provided")
        
        is_video = isinstance(images[0], Input.Video)

        if is_video:
            model_wrapper = get_sam2_video(model_path, precision)
            model_wrapper.load_wrapper()
        else:
            model_wrapper = get_sam2_image(model_path, precision)
            model_wrapper.load_wrapper()

        output_masks = []
        for idx, input in enumerate(batch_inputs):
            logger.info(f"SAM2 processing image: {idx+1}/{len(batch_inputs)}")

            image = input["image"]["value"]
            bbox_data = input["bbox_data"]["value"]

            if image is None:
                logger.warning(f"Image {idx+1} is None")
                continue
            
            if bbox_data is None:
                logger.warning(f"Bbox data {idx+1} is None")
                continue

            bbox_info = json.loads(bbox_data)
            input_boxes = cls._parse_bbox(bbox_info)

            if is_video:
                components = image.get_components()
                video_frames = components.images 

                masks = cls._segment_video(
                    model_wrapper,
                    video_frames,
                    input_boxes,
                )
                output_masks.append(masks.cpu())
            else:
                batch_masks = []
                for i in range(image.shape[0]):
                    np_image = (image[i].cpu().numpy() * 255).astype(np.uint8)
                    pil_image = Image.fromarray(np_image)

                    mask = cls._segment_image(
                        model_wrapper,
                        pil_image,
                        input_boxes,
                    )
                    batch_masks.append(mask)

                masks = torch.stack(batch_masks, dim=0).cpu()
                output_masks.append(masks)

        handle_wrapper_after_run("sam2_video" if is_video else "sam2_image")

        return io.NodeOutput(output_masks)

    @classmethod
    def _parse_bbox(cls, bbox_info: dict) -> list | None:
        if "detection" in bbox_info and bbox_info["detection"]:
            det = bbox_info["detection"]
            bbox = det.get("bbox", {})
            x0 = bbox.get("x0")
            y0 = bbox.get("y0")
            x1 = bbox.get("x1")
            y1 = bbox.get("y1")
            if all(v is not None for v in [x0, y0, x1, y1]):
                return [[[x0, y0, x1, y1]]]

        if "detections" in bbox_info and bbox_info["detections"]:
            boxes = []
            for det in bbox_info["detections"]:
                bbox = det.get("bbox", {})
                x0 = bbox.get("x0")
                y0 = bbox.get("y0")
                x1 = bbox.get("x1")
                y1 = bbox.get("y1")
                if all(v is not None for v in [x0, y0, x1, y1]):
                    boxes.append([x0, y0, x1, y1])
            if boxes:
                return [[boxes[0]]]

        return None

    @classmethod
    def _segment_image(
        cls,
        model_wrapper,
        pil_image: Image.Image,
        input_boxes: list,
    ) -> torch.Tensor:
        model = model_wrapper.model
        processor = model_wrapper.processor
        dtype = model_wrapper.dtype
        device = model_wrapper.current_device

        W, H = pil_image.size

        inputs = processor(
            images=pil_image,
            input_boxes=input_boxes,
            return_tensors="pt",
        ).to(dtype).to(device)

        outputs = model(**inputs)

        pred_masks = outputs.pred_masks.cpu()
        iou_scores = outputs.iou_scores.cpu()

        best_mask_idx = iou_scores[0, 0].argmax().item()
        best_mask = pred_masks[0, 0, best_mask_idx]

        mask_resized = torch.nn.functional.interpolate(
            best_mask.unsqueeze(0).unsqueeze(0).float(),
            size=(H, W),
            mode="bilinear",
            align_corners=False,
        )[0, 0]

        mask_tensor = (mask_resized > 0).float()

        return mask_tensor

    @classmethod
    def _segment_video(
        cls,
        model_wrapper,
        video_frames: torch.Tensor,
        input_boxes: list,
    ) -> torch.Tensor:
        model = model_wrapper.model
        processor = model_wrapper.processor
        dtype = model_wrapper.dtype
        device = model_wrapper.current_device

        frames_list = []
        for i in range(video_frames.shape[0]):
            np_frame = (video_frames[i].cpu().numpy() * 255).astype(np.uint8)
            frames_list.append(np_frame)

        inference_session = processor.init_video_session(
            video=frames_list,
            inference_device=device,
            dtype=dtype,
        )

        x0, y0, x1, y1 = input_boxes[0][0]
        input_points = [[[[x0, y0], [x1, y1]]]]
        input_labels = [[[2, 3]]]

        processor.add_inputs_to_inference_session(
            inference_session=inference_session,
            frame_idx=0,
            obj_ids=[1],
            input_points=input_points,
            input_labels=input_labels,
        )

        outputs = model(inference_session=inference_session, frame_idx=0)

        video_masks = {}
        video_height = inference_session.video_height
        video_width = inference_session.video_width

        for sam2_output in model.propagate_in_video_iterator(inference_session):
            frame_idx = sam2_output.frame_idx
            masks = processor.post_process_masks(
                [sam2_output.pred_masks],
                original_sizes=[[video_height, video_width]],
                binarize=True,
            )[0]
            video_masks[frame_idx] = masks

        num_frames = len(frames_list)
        mask_list = []
        for i in range(num_frames):
            if i in video_masks:
                mask = video_masks[i][0, 0, :, :].float()
            else:
                mask = torch.zeros(video_height, video_width)
            mask_list.append(mask)

        masks_tensor = torch.stack(mask_list, dim=0)

        return masks_tensor

    @classmethod
    def fingerprint_inputs(cls, **kwargs):
        try:
            config_mtime = os.path.getmtime(_CONFIG_FILE_PATH)
            global_config_mtime = os.path.getmtime(get_config_file_path("global"))
        except OSError:
            config_mtime = 0
            global_config_mtime = 0
            
        return hash((str(kwargs), str(config_mtime), str(global_config_mtime)))
