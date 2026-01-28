'''
@author: 
@date: 2026-01-25
@description: This node is used to generate prompts for the Qwen model.

required:
- openai
- cv2
'''
import os
import torch
import tempfile
import json
import base64
import logging
import io as bio
import numpy as np
from PIL import Image
import torch.nn.functional as F

from comfy_api.latest import io

from ..utils import utils
from ..utils.config_loader import ConfigLoader

_CONFIG_FILE_PATH = utils.get_config_file_path("qwen_llm")

class DAQwenLLMConfig(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        config = ConfigLoader(_CONFIG_FILE_PATH, strict=False)

        return io.Schema(
            node_id="DAQwenLLMConfig",
            display_name="DA QwenLLM Config",
            category="DALab/LLM/QwenLLM",
            description="Configure the QwenLLM model params at ali baiLian platform. Run first to save config.",
            is_output_node=True,
            inputs=[
                io.String.Input(
                    "api_key",
                    default=config.get("api_key", ""),
                    display_name="API Key",
                    tooltip="The API key for Qwen.From Ali BaiLian Platform.",
                ),
                io.String.Input(
                    "base_url",
                    default=config.get("base_url", "https://dashscope.aliyuncs.com/compatible-mode/v1"),
                    display_name="Base URL",
                    tooltip="The base url for Qwen.",
                ),
            ],
            outputs=[],
        )

    @classmethod
    def execute(cls, api_key, base_url) -> io.NodeOutput:
        config_data = {
            "api_key": api_key,
            "base_url": base_url,
        }
        utils.save_json(config_data, _CONFIG_FILE_PATH)
        return io.NodeOutput()

class DAQwenLLM(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        config = ConfigLoader(_CONFIG_FILE_PATH, strict=False)

        model_options = [
            "qwen-plus",
            "qwen-flash",
            "qwen3-max",
        ]
        
        return io.Schema(
            node_id="DAQwenLLM",   
            display_name="DA QwenLLM",
            category="DALab/LLM/QwenLLM",
            is_output_node=True,
            inputs=[
                io.Combo.Input(
                    "model",
                    options=model_options,
                    default=config.get("model", "qwen-plus"),
                    display_name="model",
                    tooltip="Select the Qwen model to use.",
                ),
                io.Combo.Input(
                    "prompt_type",
                    options=["default","image", "video"],
                    default="image",
                    display_name="prompt_type",
                    tooltip="Choose 'image' for static details or 'video' for motion and camera angles.",
                ),
                io.Int.Input(
                    "prompt_num",
                    default=1,
                    min=1,
                    max=20,
                    display_name="prompt_num",
                    tooltip="The number of prompts to generate.",
                ),
                io.String.Input(
                    "prompt",
                    default="one girl in a white dress, looking at the camera",
                    multiline=True,
                    display_name="prompt",
                    tooltip="Describe your idea here.",
                ),
                io.Int.Input(
                    "seed",
                    default=0,
                    min=0,
                    max=0xFFFFFFFFFFFFFFFF,
                    control_after_generate=True,
                    display_name="seed",
                    tooltip="The random seed used for creating the noise.",
                ),
                io.Boolean.Input(
                    "show_preview",
                    default=True,
                    display_name="show_preview",
                    tooltip="Whether to show the preview of the prompts.",
                ),
            ],
            outputs=[
                io.String.Output(
                    "en_prompts",
                    tooltip="The English prompts.",
                    display_name="en_prompts",
                    is_output_list=True,
                ),
                io.String.Output(
                    "zh_prompts",
                    tooltip="The Chinese prompts.",
                    display_name="zh_prompts",
                    is_output_list=True,
                ),
            ],
        )

    @classmethod
    def execute(
        cls, 
        model: str, 
        prompt_num: int, 
        prompt_type: str,
        prompt: str,
        seed: int,
        show_preview: bool,
    ) -> io.NodeOutput:
        if model == "" or prompt == "":
            raise ValueError("Model and prompt are required")
        
        config = ConfigLoader(_CONFIG_FILE_PATH, strict=False)
        
        try:
            import openai
        except ImportError:
            raise ImportError(
                "[DALab] Missing dependency: 'openai'. "
                "Please install it manually via terminal: 'pip install openai' "
                "(or 'python_embeded/python.exe -m pip install openai' if using portable ComfyUI)."
            )

        try:
            client = openai.OpenAI(
                api_key=config.get("api_key") or os.getenv("ALI_BAILIAN_API_KEY"),
                base_url=config.get("base_url") or os.getenv("ALI_BAILIAN_BASE_URL"),
            )
        except Exception as e:
            raise ValueError(f"[DALab] Failed to initialize OpenAI client: {e}")
    
        user_content = []
        if prompt_type == "video":
            system_content = SYSTEM_PROMPT_VIDEO.format(prompt_num=prompt_num)
        elif prompt_type == "image":
            system_content = SYSTEM_PROMPT_IMAGE.format(prompt_num=prompt_num)
        else:
            system_content = SYSTEM_PROMPT_DEFAULT.format(prompt_num=prompt_num)

        user_content.append({"type": "text", "text": f"User Request: {prompt}"})

        message_content = [
            {"role": "system", "content": system_content},
            {"role": "user", "content": user_content},
        ]

        try:
            logging.info(f"[DALab] QwenLLM sending request to {model}")
            completion = client.chat.completions.create(
                model=model,
                messages=message_content,
                response_format={"type": "json_object"} if prompt_type != "default" else None
            )

            content = completion.choices[0].message.content
            
            if "```json" in content:
                content = content.replace("```json", "").replace("```", "")
            content = content.strip()
            
            try:
                prompts = json.loads(content)
            except:
                raise ValueError("LLM returned invalid JSON")

            if not isinstance(prompts, list):
                 raise ValueError("LLM response must be a JSON list.")

            en_prompts = [prompt["en_prompt"] for prompt in prompts]
            zh_prompts = [prompt["zh_prompt"] for prompt in prompts]

            return io.NodeOutput(
                en_prompts, 
                zh_prompts,
                ui={"qwen_prompts": prompts} if show_preview else None
            )

        except Exception as e:
            raise ValueError(f"Error: {str(e)}")

class DAQwenVL(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        config = ConfigLoader(_CONFIG_FILE_PATH, strict=False)

        model_options = [
            "qwen3-vl-flash",
            "qwen3-vl-plus", 
        ]
        
        return io.Schema(
            node_id="DAQwenVL",   
            display_name="DA QwenVL",
            category="DALab/LLM/QwenLLM",
            is_output_node=True,
            description="Generate prompts from images or videos.",
            inputs=[
                io.Autogrow.Input(
                    "images",
                    optional=True,
                    display_name="images",
                    tooltip="The images to use as a reference for the generation.",
                    template=io.Autogrow.TemplateNames(
                        io.Image.Input(
                            "image",
                            optional=True,
                            tooltip="Up to 10 reference images."
                        ),
                        names=["image1", "image2", "image3","image4","image5","image6","image7","image8","image9","image10"],
                        min=1,
                    ),
                ),
                io.Autogrow.Input(
                    "videos",
                    optional=True,
                    display_name="videos",
                    tooltip="The videos to use as a reference for the generation.",
                    template=io.Autogrow.TemplateNames(
                        io.Video.Input(
                            "video",
                            optional=True,
                            tooltip="Up to 10 reference videos."
                        ),
                        names=["video1", "video2", "video3","video4","video5","video6","video7","video8","video9","video10"],
                        min=1,
                    ),
                ),
                io.Combo.Input(
                    "model",
                    options=model_options,
                    default=config.get("model", "qwen3-vl-flash"),
                    display_name="model",
                    tooltip="Select the Qwen model to use.",
                ),
                io.Combo.Input(
                    "prompt_type",
                    options=["default","image", "video"],
                    default="image",
                    display_name="prompt_type",
                    tooltip="Choose 'image' for static details or 'video' for motion and camera angles.",
                ),
                io.String.Input(
                    "prompt",
                    default="describe the image or video",
                    multiline=True,
                    display_name="prompt",
                    tooltip="Describe the image or video here.",
                ),
                io.Int.Input(
                    "seed",
                    default=0,
                    min=0,
                    max=0xFFFFFFFFFFFFFFFF,
                    control_after_generate=True,
                    display_name="seed",
                    tooltip="The random seed used for creating the noise.",
                ),
                io.Boolean.Input(
                    "show_preview",
                    default=True,
                    display_name="show_preview",
                    tooltip="Whether to show the preview of the prompts.",
                ),
            ],
            outputs=[
                io.String.Output(
                    "en_prompts",
                    tooltip="The English prompts.",
                    display_name="en_prompts",
                    is_output_list=True,
                ),
                io.String.Output(
                    "zh_prompts",
                    tooltip="The Chinese prompts.",
                    display_name="zh_prompts",
                    is_output_list=True,
                ),
            ],
        )

    @classmethod
    def execute(
        cls, 
        model: str, 
        prompt_type: str,
        prompt: str,
        seed: int,
        show_preview: bool,
        images = None,
        videos = None,
    ) -> io.NodeOutput:
        config = ConfigLoader(_CONFIG_FILE_PATH, strict=False)
        
        try:
            import openai
        except ImportError:
            raise ImportError(
                "[DALab] Missing dependency: 'openai'. "
                "Please install it manually via terminal: 'pip install openai' "
                "(or 'python_embeded/python.exe -m pip install openai' if using portable ComfyUI)."
            )

        try:
            client = openai.OpenAI(
                api_key=config.get("api_key") or os.getenv("ALI_BAILIAN_API_KEY"),
                base_url=config.get("base_url") or os.getenv("ALI_BAILIAN_BASE_URL"),
            )
        except Exception as e:
            raise ValueError(f"[DALab] Failed to initialize OpenAI client: {e}")

        if images is not None and len(images.values()) > 0 and videos is not None and len(videos.values()) > 0:
            raise ValueError("Only one of images or videos is allowed")
        
        if images is None and videos is None:
            raise ValueError("Choose a image or video")
    
        user_content = []
        if images is not None and len(images.values()) > 0:
            for image in images.values():
                base64_image = cls.encode_image(image)
                user_content.append({
                    "type": "image_url",
                    "image_url": {"url": base64_image}, 
                })
            if prompt_type == "video":
                system_content = SYSTEM_PROMPT_IMAGE_TO_VIDEO
            elif prompt_type == "image":
                system_content = SYSTEM_PROMPT_IMAGE_TO_IMAGE
            else:
                system_content = SYSTEM_PROMPT_IMAGE_TO_DEFAULT
        elif videos is not None and len(videos.values()) > 0:
            for video in videos.values():
                base64_video = cls.encode_video(video)
                user_content.append({
                    "type": "video_url",
                    "video_url": {"url": base64_video},
                    "fps":2
                })
            if prompt_type == "video":
                system_content = SYSTEM_PROMPT_VIDEO_TO_VIDEO
            elif prompt_type == "image":
                system_content = SYSTEM_PROMPT_VIDEO_TO_IMAGE
            else:
                system_content = SYSTEM_PROMPT_VIDEO_TO_DEFAULT
        else:
            raise ValueError("Choose a image or video")

        user_content.append({"type": "text", "text": f"User Request: {prompt}"})

        message_content = [
            {"role": "system", "content": system_content},
            {"role": "user", "content": user_content},
        ]

        try:
            logging.info(f"[DALab] QwenVL sending request to {model}")
            completion = client.chat.completions.create(
                model=model,
                messages=message_content,
                response_format={"type": "json_object"} if prompt_type != "default" else None
            )

            content = completion.choices[0].message.content
            
            if "```json" in content:
                content = content.replace("```json", "").replace("```", "")
            content = content.strip()
            
            try:
                prompts = json.loads(content)
            except:
                raise ValueError("LLM returned invalid JSON", content)

            if not isinstance(prompts, list):
                 raise ValueError("LLM response must be a JSON list.", content)

            en_prompts = [prompt["en_prompt"] for prompt in prompts]
            zh_prompts = [prompt["zh_prompt"] for prompt in prompts]

            return io.NodeOutput(
                en_prompts, 
                zh_prompts,
                ui={"qwen_prompts": prompts} if show_preview else None
            )

        except Exception as e:
            raise ValueError(f"Error: {str(e)}")

    @classmethod
    def encode_image(cls, image):
        MAX_SIZE = 1024
        image_tensor = image[0]
        
        i = 255. * image_tensor.cpu().numpy()
        img = Image.fromarray(np.clip(i, 0, 255).astype(np.uint8))

        if img.width > MAX_SIZE or img.height > MAX_SIZE:
            img.thumbnail((MAX_SIZE, MAX_SIZE), Image.Resampling.LANCZOS)
            print(f"[DALab] Downsampled image to {img.width}x{img.height}")
        
        buffered = bio.BytesIO()

        if img.mode != "RGB":
            img = img.convert("RGB")
        
        img.save(buffered, format="JPEG", quality=85)
        img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")

        base64_image = f"data:image/jpeg;base64,{img_str}"
        
        return base64_image

    @classmethod
    def encode_video(cls, video_obj):
        MAX_SIZE = 640
        
        images = video_obj.get_components().images
        fps = float(video_obj.get_components().frame_rate)
        
        if images is None:
            raise ValueError("Video has no images.")

        images = images.permute(0, 3, 1, 2)
        b, c, h, w = images.shape
        
        if h > MAX_SIZE or w > MAX_SIZE:
            scale = min(MAX_SIZE / h, MAX_SIZE / w)
            new_h = max(int(h * scale) // 2 * 2, 2)
            new_w = max(int(w * scale) // 2 * 2, 2)
            images = F.interpolate(images, size=(new_h, new_w), mode="bilinear", align_corners=False)
        
        images = images.permute(0, 2, 3, 1)

        if images.shape[-1] == 1:
            images = images.repeat(1, 1, 1, 3)
        elif images.shape[-1] == 4:
            images = images[..., :3]

        frames_numpy = (images * 255).clamp(0, 255).to(torch.uint8).cpu().numpy()

        if len(frames_numpy) > 2000:
            indices = np.linspace(0, len(frames_numpy) - 1, 2000, dtype=int)
            frames_numpy = frames_numpy[indices]
        elif len(frames_numpy) < 4:
            last = frames_numpy[-1:]
            frames_numpy = np.concatenate([frames_numpy] + [last] * (4 - len(frames_numpy)), axis=0)

        h, w = frames_numpy[0].shape[:2]
        
        with tempfile.NamedTemporaryFile(suffix='.mp4', delete=False) as tmp:
            tmp_path = tmp.name

        try:
            import cv2
        except ImportError:
            raise ImportError(
                "[DALab] Missing dependency: 'cv2'. "
                "Please install it manually via terminal: 'pip install opencv-python' "
                "(or 'python_embeded/python.exe -m pip install opencv-python' if using portable ComfyUI)."
            )
            
        try:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(tmp_path, fourcc, fps, (w, h))
            
            for frame in frames_numpy:
                frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                out.write(frame_bgr)
            
            out.release()
            
            with open(tmp_path, 'rb') as f:
                video_bytes = f.read()
        finally:
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)

        video_str = base64.b64encode(video_bytes).decode("utf-8")
        return f"data:video/mp4;base64,{video_str}"

SYSTEM_PROMPT_IMAGE_TO_DEFAULT = """
你是一位拥有敏锐观察力的 AI 视觉助手。
你的任务是：仔细观察用户提供的【参考图片】，并根据用户的【文字指令】进行回答。

【能力范围】
1. **画面描述**：如果用户要求描述图片，请详细说明画面中的主体、背景、颜色和细节。
2. **视觉问答**：准确回答关于图片内容的任何问题（如“图中有几个人？”、“这是什么花？”）。
3. **任务执行**：如果用户要求提取文字或分析数据，请精准执行。

【输出要求】
严禁 Markdown，严格遵循 JSON 数组格式：
[
    {{
        "zh_prompt": "中文描述，包含参考图的视觉特征 + 脑补的动态动作 + 运镜",
        "en_prompt": "English prompt, visual description of the reference image, plus dynamic motion (e.g. hair floating, smiling, walking), camera movement (e.g. zoom in), high quality"
    }}
]
"""

SYSTEM_PROMPT_IMAGE_TO_VIDEO = """
你是一位极具想象力的 AI 视觉导演，精通图生视频（Image-to-Video）技术。
你的任务是：观察用户提供的【参考图片】，结合用户的【文字描述】，生成于生成视频的提示词。

【处理逻辑】
1. **视觉解析**：精准描述参考图片的主体、环境、光影和风格。
2. **动态脑补**：这是最关键的一步！根据画面内容，想象合理的运动。
   - 如果是人物：描述微表情、眨眼、头发随风飘动、肢体舒展。
   - 如果是风景：描述云的流动、水的波纹、光影的变化。
3. **运镜设计**：添加适合该画面的摄像机运动（如 Slow zoom in, Pan right, Static shot with dynamics）。

【输出要求】
严禁 Markdown，严格遵循 JSON 数组格式：
[
    {{
        "zh_prompt": "中文描述，包含参考图的视觉特征 + 脑补的动态动作 + 运镜",
        "en_prompt": "English prompt, visual description of the reference image, plus dynamic motion (e.g. hair floating, smiling, walking), camera movement (e.g. zoom in), high quality"
    }}
]
"""

SYSTEM_PROMPT_IMAGE_TO_IMAGE = """
你是一位资深的 AI 艺术鉴赏家和提示词专家，精通图生图（Image-to-Image）重绘技术。
你的任务是：观察用户提供的【参考图片】，结合用户的【文字描述】，生成高质量的静态图片提示词。

【处理逻辑】
1. **深度解析**：详细拆解参考图片的构图（Composition）、色彩（Color Palette）、材质（Texture）和光影（Lighting）。
2. **风格融合**：如果用户指定了新风格，请将参考图的内容与新风格完美融合；如果未指定，则保持参考图的高画质风格。
3. **画质强化**：加入 8k, photorealistic, intricate details, masterpiece 等高质量修饰词。

【输出要求】
严禁 Markdown，严格遵循 JSON 数组格式：
[
    {{
        "zh_prompt": "中文描述，详细描写参考图的内容、构图和光影细节",
        "en_prompt": "English prompt, highly detailed description of the reference image, specific texture, lighting, composition, 8k, masterpiece"
    }}
]
"""

SYSTEM_PROMPT_VIDEO_TO_DEFAULT = """
你是一位专业的 AI 视频内容分析师。
你的任务是：观看用户提供的【参考视频】，理解视频的时间流逝和事件发展，并根据用户的【文字指令】进行回答。

【能力范围】
1. **内容总结**：概括视频中发生了什么故事或事件。
2. **细节捕捉**：回答关于视频中特定动作、物体或环境细节的问题。
3. **动态分析**：描述视频的氛围、节奏或运镜方式。

【输出要求】
严禁 Markdown，严格遵循 JSON 数组格式：
[
    {{
        "zh_prompt": "中文描述，包含参考图的视觉特征 + 脑补的动态动作 + 运镜",
        "en_prompt": "English prompt, visual description of the reference image, plus dynamic motion (e.g. hair floating, smiling, walking), camera movement (e.g. zoom in), high quality"
    }}
]
"""

SYSTEM_PROMPT_VIDEO_TO_VIDEO = """
你是一位专业的 AI 动作捕捉师和电影导演。
你的任务是：观察用户提供的【参考视频】，结合用户的【文字描述】，生成于生成视频的提示词，旨在重现或风格化该视频片段。

【处理逻辑】
1. **动作捕捉**：精准描述视频中主体的连续动作（如 "一个女孩在转身回头" 或 "汽车在雨中飞驰"）。
2. **节奏与连贯性**：描述视频的速度感（Slow motion / Timelapse）和物理流动。
3. **风格迁移**：保留原视频的【动作】和【构图】，但根据用户文字应用新的【风格】（如将实拍视频描述为赛博朋克动画风格）。

【输出要求】
严禁 Markdown，严格遵循 JSON 数组格式：
[
    {{
        "zh_prompt": "中文描述，重点在于准确还原参考视频的动作轨迹和运镜",
        "en_prompt": "English prompt, precise description of the action sequence from reference video, camera movement, continuity, high quality"
    }}
]
"""

SYSTEM_PROMPT_VIDEO_TO_IMAGE = """
你是一位专业的剧照摄影师。
你的任务是：观察用户提供的【参考视频】，从中提炼出最核心的视觉画面，生成高质量的静态图片提示词。。

【处理逻辑】
1. **瞬间凝固**：不要描述连续动作（如 "开始跑然后停下"），而是描述一个精彩的【定格瞬间】（如 "奔跑中悬空的瞬间"）。
2. **画质提升**：视频通常带有动态模糊，你的提示词必须强调“清晰”、“锐利”、“高分辨率”，以生成高质量的静帧。
3. **氛围营造**：捕捉视频中的光影氛围和情绪。

【输出要求】
严禁 Markdown，严格遵循 JSON 数组格式：
[
    {{
        "zh_prompt": "中文描述，描述视频中定格的精彩瞬间，强调清晰度和构图",
        "en_prompt": "English prompt, a frozen moment from the video, cinematic still, sharp focus, highly detailed, 8k, masterpiece"
    }}
]
"""

SYSTEM_PROMPT_DEFAULT = """
任务是根据用户的输入，生成 **{prompt_num}** 组高质量的答案。
如果用户让你优化提示词，请直接输出优化后的结果；
如果用户问你问题，请直接回答，请保持回答简洁、专业。
每组答案需要分别输出中文和英文的两种语言的结果。

【输出要求】
严禁 Markdown，严格遵循 JSON 数组格式：
[
    {{
        "zh_prompt": "中文描述描述",
        "en_prompt": "English prompt, description"
    }}
]
"""

SYSTEM_PROMPT_IMAGE = """
你是一位精通 Stable Diffusion 和 Flux 的 AI 视觉艺术导演。
任务是根据用户的输入，生成 **{prompt_num}** 组高质量的静态图片提示词。

【生成要求】
1. **画面质感**：必须强调光影 (Cinematic lighting, Volumetric)、材质 (Texture)、画质 (8k, Masterpiece) 和构图 (Wide angle, Macro)。
2. **静态美学**：关注画面定格瞬间的细节描写。
3. **中英对照**：英文提示词是生成的关键，请使用 AI 社区通用的高质量 Tag 或自然语言。

【输出格式】
必须严格遵循以下 JSON 数组格式，严禁使用 Markdown 代码块：
[
    {{
        "zh_prompt": "中文描述，包含丰富的画面细节",
        "en_prompt": "English prompt, highly detailed, photorealistic, cinematic lighting, 8k, masterpiece"
    }},
]
"""

SYSTEM_PROMPT_VIDEO = """
你是一位精通 Sora, Runway 和 Kling 的 AI 电影导演。
任务是根据用户的输入，生成 **{prompt_num}** 组高质量的视频片段提示词。

【生成要求】
1. **动态强化**：必须包含具体的动作描述 (Walking, Running, Flowing) 和物理规律。
2. **运镜语言**：必须包含专业的摄像机运动描述 (Camera pan right, Zoom in, Drone shot, Static camera)。
3. **连贯性**：描述物体如何移动、时间如何流逝。
4. **中英对照**：英文提示词请使用自然语言的长句描述，强调动作流畅。

【输出格式】
必须严格遵循以下 JSON 数组格式，严禁使用 Markdown 代码块：
[
    {{
        "zh_prompt": "中文描述，包含具体的动作和运镜方式",
        "en_prompt": "Cinematic shot, camera pans right following the subject, slow motion, dynamic movement, high quality"
    }},
]
"""