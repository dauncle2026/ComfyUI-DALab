'''
@author:
@date: 2026-01-25
@description: This node is used to generate captions or grounding results using Florence2.

required:
- timm
'''
import os
import json
import numpy as np
from PIL import Image, ImageDraw, ImageFont

import folder_paths
from comfy_api.latest import io, Input, ui

from ..utils import utils
from ..utils.config_loader import ConfigLoader
from ..utils.logger import logger
from ..utils.paths import get_config_file_path
from ..utils.wrappers.florence2 import get_florence2
from ..utils.wrappers.base import handle_wrapper_after_run

_CONFIG_FILE_PATH = get_config_file_path("florence2")

TASK_PROMPTS = {
    'short': '<CAPTION>',
    'middle': '<DETAILED_CAPTION>',
    'long': '<MORE_DETAILED_CAPTION>',

    'use_text': '<CAPTION_TO_PHRASE_GROUNDING>',
    'use_tag': '<OPEN_VOCABULARY_DETECTION>',

    'object': '<OD>',
    'object_with_detail': '<DENSE_REGION_CAPTION>',
}

COLORMAP = [
    'blue', 'orange', 'green', 'purple', 'brown', 'pink',
    'olive', 'cyan', 'red', 'lime', 'indigo', 'violet',
    'aqua', 'magenta', 'gold', 'tan', 'skyblue'
]

class DAFlorence2Config(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        config = ConfigLoader(_CONFIG_FILE_PATH, strict=False)
        model_path = os.path.join(folder_paths.models_dir, "dalab", "florence2", "Florence-2-large-ft")

        return io.Schema(
            node_id="DAFlorence2Config",
            display_name="DA Florence2 Config",
            category="DALab/Tools/Florence2",
            description="Configure Florence2 model settings. Run first to save config.",
            is_output_node=True,
            inputs=[
                io.String.Input(
                    "model_path",
                    default=config.get("model_path", model_path),
                    display_name="Model Path",
                    tooltip="Path to Florence2 model directory.",
                ),
                io.Combo.Input(
                    "precision",
                    default=config.get("precision", "fp16"),
                    options=["fp16", "bf16", "fp32"],
                    display_name="Precision",
                    tooltip="Model precision. fp16 recommended for most GPUs.",
                ),
                io.Int.Input(
                    "max_new_tokens",
                    default=config.get("max_new_tokens", 1024),
                    min=64,
                    max=4096,
                    display_name="Max New Tokens",
                    tooltip="Maximum number of tokens to generate.",
                ),
                io.Int.Input(
                    "num_beams",
                    default=config.get("num_beams", 3),
                    min=1,
                    max=16,
                    display_name="Num Beams",
                    tooltip="Number of beams for beam search.",
                ),
                io.Boolean.Input(
                    "do_sample",
                    default=config.get("do_sample", False),
                    display_name="Do Sample",
                    tooltip="Whether to use sampling for generation.",
                ),
                io.Int.Input(
                    "bbox_line_width",
                    default=config.get("bbox_line_width", 3),
                    min=1,
                    max=10,
                    display_name="Bbox Line Width",
                    tooltip="Line width for bounding box drawing.",
                ),
                io.Int.Input(
                    "font_size",
                    default=config.get("font_size", 16),
                    min=8,
                    max=48,
                    display_name="Font Size",
                    tooltip="Font size for labels.",
                ),
            ],
            outputs=[],
        )

    @classmethod
    def execute(
        cls,
        model_path: str,
        precision: str,
        max_new_tokens: int,
        num_beams: int,
        do_sample: bool,
        bbox_line_width: int,
        font_size: int,
    ) -> io.NodeOutput:
        config_data = {
            "model_path": model_path,
            "precision": precision,
            "max_new_tokens": max_new_tokens,
            "num_beams": num_beams,
            "do_sample": do_sample,
            "bbox_line_width": bbox_line_width,
            "font_size": font_size,
        }
        utils.save_json(config_data, _CONFIG_FILE_PATH)
        return io.NodeOutput()

class DAFlorence2(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        task_options = [
            io.DynamicCombo.Option("caption", [
                io.Combo.Input(
                    "task",
                    default="short",
                    options=["short","middle","long"],
                    display_name="task",
                    tooltip="Select the Florence2 task type.",
                ),
            ]),
            io.DynamicCombo.Option("bbox_by_text", [
                io.Combo.Input(
                    "task",
                    default="use_text",
                    options=["use_text","use_tag"],
                    display_name="task",
                    tooltip="Select the Florence2 task type.",
                ),
                io.String.Input(
                    "prompts",
                    default="",
                    multiline=True,
                    tooltip="Text input for phrase grounding task.",
                ),
                io.Int.Input(
                    "bbox_index",
                    default=-1,
                    min=-1,
                    max=100,
                    display_name="bbox_index",
                    tooltip="Index of the bounding box to choose.",
                ),
            ]),
            io.DynamicCombo.Option("bbox_by_index", [
                io.Combo.Input(
                    "task",
                    default="object",
                    options=["object","object_with_detail"],
                    display_name="task",
                    tooltip="Select the Florence2 task type.",
                ),
                io.Int.Input(
                    "bbox_index",
                    default=-1,
                    min=-1,
                    max=100,
                    display_name="bbox_index",
                    tooltip="Index of the bounding box to choose.",
                ),
            ]),
        ]

        return io.Schema(
            node_id="DAFlorence2",
            display_name="DA Florence2",
            category="DALab/Tools/Florence2",
            description="Florence2 for image captioning and phrase grounding.",
            is_input_list=True,
            is_output_node=True,
            inputs=[
                io.MultiType.Input(
                    "images",
                    types=[io.Image, io.Video],
                    tooltip="Input images or video for Florence2 processing.If video first frame will be used for processing.",
                    display_name="images",
                ),
                io.DynamicCombo.Input(
                    "task_options",
                    options=task_options,
                    display_name="task_options",
                    tooltip="Select the Florence2 task type.",
                ),
                io.Int.Input(
                    "seed",
                    default=0,
                    min=0,
                    max=0xFFFFFFFFFFFFFFFF,
                    display_name="seed",
                    tooltip="Seed for the Florence2 processing.",
                    control_after_generate=True,
                ),
            ],
            outputs=[
                io.String.Output(
                    "prompts",
                    tooltip="Generated captions or grounding results.",
                    is_output_list=True,
                ),
                io.Custom("BBOX_DATA").Output(
                    "bbox_data",
                    tooltip="Bounding box data in JSON format.",
                    is_output_list=True,
                ),
            ],
        )

    @classmethod
    def execute(
        cls,
        images: list,
        task_options: list[dict],
        seed: list[int],
    ) -> io.NodeOutput:
        config = ConfigLoader(_CONFIG_FILE_PATH, strict=True)

        batch_inputs = utils.inputs_to_batch(
            image=images,
            nested_inputs={
                "task_option": task_options,
            },
        )

        model_path = config.get("model_path")
        precision = config.get("precision")
        max_new_tokens = config.get("max_new_tokens")
        num_beams = config.get("num_beams")
        do_sample = config.get("do_sample")
        bbox_line_width = config.get("bbox_line_width")
        font_size = config.get("font_size")

        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Florence2 model not found.Please download model from https://huggingface.co/microsoft/Florence-2-large-ft to 'models/dalab/florence2/Florence-2-large-ft'")
        
        config_path = os.path.join(model_path, "config.json")
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Florence2 model not found.Please download model from https://huggingface.co/microsoft/Florence-2-large-ft to 'models/dalab/florence2/Florence-2-large-ft'")
        

        model_wrapper = get_florence2(model_path, precision)
        model_wrapper.load_wrapper()

        model = model_wrapper.model
        processor = model_wrapper.processor
        dtype = model_wrapper.dtype
        device = model_wrapper.current_device

        output_prompts = []
        output_bbox_data = []
        output_images = []
        for idx, input in enumerate(batch_inputs):
            logger.info(f"Florence2 processing image: {idx+1}/{len(batch_inputs)}")
            image = input["image"]["value"]
            options = input["task_option"]
            task_type = options["task_options"]["value"]

            if image is None:
                logger.warning(f"Image {idx+1} is None")
                continue

            if isinstance(image, Input.Video):
                components = image.get_components()
                frame_images = components.images
                np_image = (frame_images[0].cpu().numpy() * 255).astype(np.uint8)
            else:
                if len(image.shape) == 4:
                    np_image = (image[0].cpu().numpy() * 255).astype(np.uint8)
                else:
                    np_image = (image.cpu().numpy() * 255).astype(np.uint8)
            
            pil_image = Image.fromarray(np_image)
            W, H = pil_image.size

            task_name = TASK_PROMPTS.get(options["task"]["value"], '<MORE_DETAILED_CAPTION>')
            if task_type == "bbox_by_text":
                prompt = task_name + " " + options["prompts"]["value"]
            else:
                prompt = task_name

            inputs = processor(
                text=prompt,
                images=pil_image,
                return_tensors="pt",
                do_rescale=False
            ).to(dtype).to(device)

            model_outputs = model.generate(
                input_ids=inputs["input_ids"],
                pixel_values=inputs["pixel_values"],
                max_new_tokens=max_new_tokens,
                do_sample=do_sample,
                num_beams=num_beams,
                use_cache=False,
            )

            decoded_results = processor.batch_decode(model_outputs, skip_special_tokens=False)[0]
            processed_results = processor.post_process_generation(
                decoded_results, task=task_name, image_size=(W, H)
            )

            clean_results = decoded_results.replace('</s>', '').replace('<s>', '')
            for tag in TASK_PROMPTS.values():
                clean_results = clean_results.replace(tag, '')
            clean_results = clean_results.strip()

            if task_type == "caption":
                output_prompts.append(clean_results)
                output_bbox_data.append("{}")
                output_images.append(pil_image)
            elif task_type == "bbox_by_text":
                output_prompts.append(clean_results)

                grounding_result = processed_results.get(task_name, {})
                annotated_image, bbox_info = cls._draw_bbox(
                    pil_image, grounding_result, options["bbox_index"]["value"], bbox_line_width, font_size
                )
                output_images.append(annotated_image)
                output_bbox_data.append(json.dumps(bbox_info))
            elif task_type == "bbox_by_index":
                output_prompts.append(clean_results)

                grounding_result = processed_results.get(task_name, {})
                annotated_image, bbox_info = cls._draw_bbox(
                    pil_image, grounding_result, options["bbox_index"]["value"], bbox_line_width, font_size
                )
                output_images.append(annotated_image)
                output_bbox_data.append(json.dumps(bbox_info))

        if len(output_images) > 0:
            ui_images = utils.save_image_to_file(output_images)
        else:
            ui_images = []

        handle_wrapper_after_run("florence2")

        return io.NodeOutput(output_prompts, output_bbox_data,ui=ui.SavedImages(ui_images))

    @classmethod
    def _draw_bbox(
        cls,
        image: Image.Image,
        grounding_result: dict,
        selected_index: int = 0,
        line_width: int = 3,
        font_size: int = 16
    ) -> tuple[Image.Image, dict]:
        annotated_image = image.copy()
        draw = ImageDraw.Draw(annotated_image)

        try:
            font = ImageFont.truetype("DejaVuSans.ttf", font_size)
        except:
            try:
                font = ImageFont.load_default(size=font_size)
            except:
                font = ImageFont.load_default()

        bboxes = grounding_result.get('bboxes', [])
        labels = grounding_result.get('labels', [])

        W, H = image.size
        bbox_info = {
            'image_size': {'width': W, 'height': H},
            'selected_index': selected_index,
            'detection': None,
            'detections': []
        }

        if not bboxes:
            return annotated_image, bbox_info

        # Determine indices to process
        if selected_index == -1:
            indices = range(len(bboxes))
        elif selected_index < len(bboxes):
            indices = [selected_index]
        else:
            return annotated_image, bbox_info

        for idx in indices:
            bbox = bboxes[idx]
            label = labels[idx] if idx < len(labels) else ""
            color = COLORMAP[idx % len(COLORMAP)]

            x0, y0, x1, y1 = bbox
            if x1 < x0:
                x0, x1 = x1, x0
            if y1 < y0:
                y0, y1 = y1, y0

            draw.rectangle([x0, y0, x1, y1], outline=color, width=line_width)

            indexed_label = f"{idx}"
            text_bbox = draw.textbbox((0, 0), indexed_label, font=font)
            text_width = text_bbox[2] - text_bbox[0]
            text_height = text_bbox[3] - text_bbox[1]

            text_x = x0
            text_y = y0

            if text_y < 0:
                text_y = y1 + 2
            if text_x + text_width > W:
                text_x = W - text_width
            if text_x < 0:
                text_x = 0

            draw.rectangle(
                [text_x, text_y, text_x + text_width + 4, text_y + text_height + 4],
                fill=color
            )
            draw.text((text_x + 2, text_y), indexed_label, fill='white', font=font)

            detection_info = {
                'index': idx,
                'label': label,
                'bbox': {
                    'x0': x0, 'y0': y0, 'x1': x1, 'y1': y1,
                    'normalized': {
                        'x0': x0 / W, 'y0': y0 / H,
                        'x1': x1 / W, 'y1': y1 / H
                    }
                }
            }

            if selected_index == -1:
                bbox_info['detections'].append(detection_info)
            else:
                bbox_info['detection'] = detection_info

        return annotated_image, bbox_info

    @classmethod
    def fingerprint_inputs(cls, **kwargs):
        try:
            config_mtime = os.path.getmtime(_CONFIG_FILE_PATH)
            global_config_mtime = os.path.getmtime(get_config_file_path("global"))
        except:
            config_mtime = 0
            global_config_mtime = 0

        return hash((str(kwargs), str(config_mtime), str(global_config_mtime)))
