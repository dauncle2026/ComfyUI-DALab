import os
import torch
import math
import torchaudio
from fractions import Fraction
import logging

import folder_paths
from comfy_api.latest import io, ui, Types,InputImpl

from ..utils import utils
from ..utils.feishu_manager import FeishuManager
from ..utils.config_loader import ConfigLoader
from ..utils.paths import get_config_file_path
from ..utils.logger import logger

_CONFIG_FILE_PATH = get_config_file_path("feishu")

class DASaveImage(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:

        save_to_feishu_options = [
            io.DynamicCombo.Option("no", [
                io.String.Input(
                    "filename_prefix",
                    default="ComfyUI",
                    tooltip="The filename prefix for the images. Default: ComfyUI",
                ),
            ]),
            io.DynamicCombo.Option("yes", [
                io.String.Input(
                    "filename_prefix",
                    default="feishu/image",
                    tooltip="The filename prefix for the images. Default: ComfyUI",
                ),
                io.String.Input(
                    "feishu_image_field_name",
                    default="image1",
                    tooltip="The feishu image field name. Default: image1",
                ),
                io.Custom("FEISHU_RECORD_IDS").Input(
                    "feishu_record_ids",
                    tooltip="The feishu record ids.",
                ),
                io.Custom("FEISHU_CONFIG").Input(
                    "feishu_config",
                    tooltip="The feishu config.",
                ),
            ]),
        ]

        return io.Schema(
            node_id="DASaveImage",  
            display_name="DA Save Image",
            category="DALab/Tools/File",
            description="Save the images to the file.Can save to local file or feishu table.",
            is_input_list=True,
            inputs=[
                io.Image.Input("images"),
                io.DynamicCombo.Input(
                    "save_to_feishu",
                    options=save_to_feishu_options,
                    display_name="save to feishu",
                    tooltip="Whether to save the images to Feishu.",
                ),
            ],
            outputs=[
                io.Image.Output(
                    "images",
                    tooltip="The original images.",
                    display_name="images",
                    is_output_list=True,
                ),
            ],
            is_output_node=True,
        )

    @classmethod
    def execute(
        cls, 
        images: list[io.Image.Type],
        save_to_feishu: list[str],
    ) -> io.NodeOutput:
        filename_prefix = save_to_feishu['filename_prefix'][0]

        config = ConfigLoader(_CONFIG_FILE_PATH, strict=True)
        ext = config.get("image_local_save_ext", ".png")

        results = utils.save_image_tensor_to_file(images, filename_prefix=filename_prefix, ext=ext)

        if save_to_feishu['save_to_feishu'][0] == "yes":
            image_field_name = save_to_feishu['feishu_image_field_name'][0]
            
            app_token = save_to_feishu['feishu_config'][0]['app_token']
            table_id = save_to_feishu['feishu_config'][0]['table_id']
            record_ids = save_to_feishu['feishu_record_ids']

            if app_token is None or table_id is None or record_ids is None or len(record_ids) == 0:
                raise ValueError("app token, table id and record ids are required")
            
            FeishuManager(config).batch_update_records(app_token, table_id, results, record_ids, image_field_name)
        else:
            pass

        return io.NodeOutput(
            images,
            ui=ui.SavedImages(results)
        )

class DASaveVideo(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        save_to_feishu_options = [
            io.DynamicCombo.Option("no", [
                io.String.Input(
                    "filename_prefix",
                    default="video/ComfyUI",
                    tooltip="The filename prefix for the videos. Default: video/ComfyUI",
                ),
            ]),
            io.DynamicCombo.Option("yes", [
                io.String.Input(
                    "filename_prefix",
                    default="feishu/video",
                    tooltip="The filename prefix for the videos. Default: feishu/video",
                ),
                io.String.Input(
                    "feishu_video_field_name",
                    default="video1",
                    tooltip="The feishu video field name. Default: video1",
                ),
                io.Custom("FEISHU_RECORD_IDS").Input(
                    "feishu_record_ids",
                    tooltip="The feishu record ids.",
                ),
                io.Custom("FEISHU_CONFIG").Input(
                    "feishu_config",
                    tooltip="The feishu config.",
                ),
            ]),
        ]

        return io.Schema(
            node_id="DASaveVideo",
            display_name="DA Save Video",
            category="DALab/Tools/File",
            description="Save the videos to the file.Can save to local file or feishu table.",
            is_input_list=True,
            inputs=[
                io.Video.Input("videos"),
                io.DynamicCombo.Input(
                    "save_to_feishu",
                    options=save_to_feishu_options,
                    display_name="save to feishu",
                    tooltip="Whether to save the videos to Feishu.",
                ),
            ],
            outputs=[
                io.Video.Output(
                    "videos",
                    tooltip="The original videos.",
                    display_name="videos",
                    is_output_list=True,
                ),
            ],
            is_output_node=True,
        )

    @classmethod
    def execute(
        cls, 
        videos: list[io.Video.Type],
        save_to_feishu: list[str],
    ) -> io.NodeOutput:
        filename_prefix = save_to_feishu['filename_prefix'][0]

        config = ConfigLoader(_CONFIG_FILE_PATH, strict=True)

        results = []
        for video in videos:
            width, height = video.get_dimensions()

            if width % 2 != 0 or height % 2 != 0:
                new_width = width - (width % 2)
                new_height = height - (height % 2)
                logger.info(f"Auto-cropping video from {width}x{height} to {new_width}x{new_height}")

                video_components = video.get_components()
                cropped_images = video_components.images[:, :new_height, :new_width, :]
                video = InputImpl.VideoFromComponents(
                    Types.VideoComponents(
                        images=cropped_images,
                        audio=video_components.audio,
                        frame_rate=video_components.frame_rate
                    )
                )
                width, height = new_width, new_height

            full_output_folder, filename, counter, subfolder, filename_prefix = folder_paths.get_save_image_path(
                filename_prefix,
                folder_paths.get_output_directory(),
                width,
                height
            )

            file = f"{filename}_{counter:05}_.{Types.VideoContainer.get_extension('auto')}"

            video.save_to(
                os.path.join(full_output_folder, file),
                format=Types.VideoContainer("auto"),
                codec="auto",
                metadata=None
            )
            
            results.append({
                "filename": file,
                "subfolder": subfolder,
                "type": "output",
                "format": "video/mp4",
                "width": width,
                "height": height
            })
            
            counter += 1

        if save_to_feishu['save_to_feishu'][0] == "yes":
            video_field_name = save_to_feishu['feishu_video_field_name'][0]
            
            app_token = save_to_feishu['feishu_config'][0]['app_token']
            table_id = save_to_feishu['feishu_config'][0]['table_id']
            record_ids = save_to_feishu['feishu_record_ids']

            if app_token is None or table_id is None or record_ids is None or len(record_ids) == 0:
                raise ValueError("app token, table id and record ids are required")

            FeishuManager(config).batch_update_records(app_token, table_id, results, record_ids, video_field_name)
        else:
            pass

        return io.NodeOutput(videos, ui={"custom_videos": results})


class DAConcatVideo(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        save_to_feishu_options = [
            io.DynamicCombo.Option("no", [
                io.String.Input(
                    "filename_prefix",
                    default="video/ComfyUI",
                    tooltip="The filename prefix for the videos. Default: video/ComfyUI",
                ),
            ]),
            io.DynamicCombo.Option("yes", [
                io.String.Input(
                    "filename_prefix",
                    default="feishu/video",
                    tooltip="The filename prefix for the videos. Default: feishu/video",
                ),
                io.String.Input(
                    "feishu_video_field_name",
                    default="video1",
                    tooltip="The feishu video field name. Default: video1",
                ),
                io.Custom("FEISHU_RECORD_IDS").Input(
                    "feishu_record_ids",
                    tooltip="The feishu record ids.",
                ),
                io.Custom("FEISHU_CONFIG").Input(
                    "feishu_config",
                    tooltip="The feishu config.",
                ),
            ]),
        ]

        return io.Schema(
            node_id="DAConcatVideo",
            display_name="DA Concat Video",
            category="DALab/Tools/File",
            description="Concat the videos to the file.Can save to local file or feishu table.",
            is_input_list=True,
            inputs=[
                io.Autogrow.Input(
                    id="videos",
                    optional=True,
                    template=io.Autogrow.TemplateNames(
                        io.Video.Input("videos"),
                        names=["video1", "video2", "video3", "video4", "video5", "video6", "video7", "video8", "video9", "video10"],
                        min=1,
                    ),
                ),
                io.DynamicCombo.Input(
                    "save_to_feishu",
                    options=save_to_feishu_options,
                    display_name="save to feishu",
                    tooltip="Whether to save the videos to Feishu.",
                ),
            ],
            outputs=[
                io.Video.Output(
                    "videos",
                    tooltip="The original videos.",
                    display_name="videos",
                    is_output_list=True,
                ),
            ],
            is_output_node=True,
        )

    @classmethod
    def execute(
        cls, 
        videos: list[io.Video.Type],
        save_to_feishu: list[str],
    ) -> io.NodeOutput:
        filename_prefix = save_to_feishu['filename_prefix'][0]

        batch_inputs = utils.inputs_to_batch(
            nested_inputs={
                "videos": videos,
            }
        )

        config = ConfigLoader(_CONFIG_FILE_PATH, strict=True)

        results = []
        for idx, inputs in enumerate(batch_inputs):
            logger.info(f"Process video batch {idx+1} of {len(batch_inputs)}")
            output_images = []
            output_audio_waveform = []
            counter = 0

            for i,video_input in enumerate(inputs["videos"].values()):
                logger.info(f"Concat video batch:{idx+1} part: {i+1}/{len(inputs['videos'].values())}")

                if video_input["value"] is None:
                    logger.info(f"Concat video batch:{idx+1} part: {i+1}/{len(inputs['videos'].values())} is None")
                    continue
                else:
                    video = video_input["value"]
                
                video_components = video.get_components()
                video_images = video_components.images
                video_height = video_images.shape[1]
                video_width = video_images.shape[2]
                video_fps = video_components.frame_rate

                if video_components.audio is not None:
                    audio_waveform = video_components.audio["waveform"]
                    audio_sample_rate = video_components.audio["sample_rate"]
                else:
                    audio_waveform = None
                    audio_sample_rate = None
                
                if len(output_images) == 0:
                    full_output_folder, filename, counter, subfolder, filename_prefix = folder_paths.get_save_image_path(
                        filename_prefix,
                        folder_paths.get_output_directory(),
                        video_width,
                        video_height
                    )

                    target_fps = video_fps
                    target_width = video_width - (video_width % 2)
                    target_height = video_height - (video_height % 2)
                    if target_width != video_width or target_height != video_height:
                        logger.info(f"Auto-adjusting target size from {video_width}x{video_height} to {target_width}x{target_height}")

                    if audio_waveform is not None:
                        target_audio_sample_rate = audio_sample_rate
                    else:
                        target_audio_sample_rate = 44100
                
                if video_fps != target_fps:
                    logger.info(
                        f"Resample video batch:{idx+1} part: {i+1}/{len(inputs['videos'].values())} from {video_fps} to {target_fps}"
                    )
                    video_images = utils.resample_video_tensor(video_images, video_fps, target_fps)
                
                if video_width != target_width or video_height != target_height:
                    logger.info(
                        f"Scale video batch:{idx+1} part: {i+1}/{len(inputs['videos'].values())} from {video_width}x{video_height} to {target_width}x{target_height}"
                    )
                    video_images = utils.scale_by_width_height(video_images, target_width, target_height, "bilinear", "center")

                if audio_waveform is not None:
                    if audio_waveform.shape[1] == 1:
                        audio_waveform = audio_waveform.repeat(1, 2, 1)
                        logger.info(
                            f"Convert mono audio batch:{idx+1} part: {i+1}/{len(inputs['videos'].values())} to stereo"
                        )

                    if audio_sample_rate != target_audio_sample_rate:
                        logger.info(
                            f"Resample audio batch:{idx+1} part: {i+1}/{len(inputs['videos'].values())} from {audio_sample_rate} to {target_audio_sample_rate}"
                        )
                        audio_waveform = torchaudio.functional.resample(
                            audio_waveform, audio_sample_rate, target_audio_sample_rate
                        )
                else:
                    logger.info(
                        f"Create empty audio batch:{idx+1} part: {i+1}/{len(inputs['videos'].values())} with sample rate {target_audio_sample_rate}"
                    )
                    audio_waveform = torch.zeros((1,2,math.ceil(len(video_images)/target_fps * target_audio_sample_rate)))

                output_images.append(video_images)
                output_audio_waveform.append(audio_waveform)

            video_components = InputImpl.VideoFromComponents(
                Types.VideoComponents(
                    images=torch.cat(output_images, dim=0), 
                    audio={
                        "waveform": torch.cat(output_audio_waveform, dim=2), 
                        "sample_rate": target_audio_sample_rate
                    },
                    frame_rate=Fraction(target_fps)
                )
            )

            file = f"{filename}_{counter:05}_.{Types.VideoContainer.get_extension('auto')}"
            
            video_components.save_to(
                os.path.join(full_output_folder, file),
                format=Types.VideoContainer("auto"),
                codec="auto",
                metadata=None
            )
            
            results.append({
                "filename": file,
                "subfolder": subfolder,
                "type": "output",
                "format": "video/mp4",
                "width": target_width,
                "height": target_height
            })
            
            counter += 1

        if save_to_feishu['save_to_feishu'][0] == "yes":
            video_field_name = save_to_feishu['feishu_video_field_name'][0]
            
            app_token = save_to_feishu['feishu_config'][0]['app_token']
            table_id = save_to_feishu['feishu_config'][0]['table_id']
            record_ids = save_to_feishu['feishu_record_ids']

            if app_token is None or table_id is None or record_ids is None or len(record_ids) == 0:
                raise ValueError("app token, table id and record ids are required")

            FeishuManager(config).batch_update_records(app_token, table_id, results, record_ids, video_field_name)
        else:
            pass

        return io.NodeOutput(videos, ui={"custom_videos": results})


class DASaveAudio(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        save_to_feishu_options = [
            io.DynamicCombo.Option("no", [
                io.String.Input(
                    "filename_prefix",
                    default="audio/ComfyUI",
                    tooltip="The filename prefix for the audio. Default: audio/ComfyUI",
                ),
            ]),
            io.DynamicCombo.Option("yes", [
                io.String.Input(
                    "filename_prefix",
                    default="feishu/audio",
                    tooltip="The filename prefix for the audio. Default: feishu/audio",
                ),
                io.String.Input(
                    "feishu_audio_field_name",
                    default="audio1",
                    tooltip="The feishu audio field name. Default: audio1",
                ),
                io.Custom("FEISHU_RECORD_IDS").Input(
                    "feishu_record_ids",
                    tooltip="The feishu record ids.",
                ),
                io.Custom("FEISHU_CONFIG").Input(
                    "feishu_config",
                    tooltip="The feishu config.",
                ),
            ]),
        ]

        return io.Schema(
            node_id="DASaveAudio",
            display_name="DA Save Audio",
            category="DALab/Tools/File",
            description="Save the audio to the file.Can save to local file or feishu table.",
            is_input_list=True,
            inputs=[
                io.Audio.Input("audios"),
                io.DynamicCombo.Input(
                    "save_to_feishu",
                    options=save_to_feishu_options,
                    display_name="save to feishu",
                    tooltip="Whether to save the audio to Feishu.",
                ),
            ],
            outputs=[
                io.Audio.Output(
                    "audios",
                    tooltip="The original audio.",
                    display_name="audios",
                    is_output_list=True,
                ),
            ],
            is_output_node=True,
        )

    @classmethod
    def execute(
        cls, 
        audios: list[io.Audio.Type],
        save_to_feishu: list[str],
    ) -> io.NodeOutput:
        filename_prefix = save_to_feishu['filename_prefix'][0]

        config = ConfigLoader(_CONFIG_FILE_PATH, strict=True)

        results = []
        for audio in audios:
            save_result = ui.AudioSaveHelper.save_audio(
                audio,
                filename_prefix=filename_prefix,
                folder_type=io.FolderType.output,
                cls=cls,
            )
            
            results.append({
                "filename": save_result[0]["filename"],
                "subfolder": save_result[0]["subfolder"],
                "type": save_result[0]["type"],
                "format": "flac",
            })
        
        if save_to_feishu['save_to_feishu'][0] == "yes":
            audio_field_name = save_to_feishu['feishu_audio_field_name'][0]
            
            app_token = save_to_feishu['feishu_config'][0]['app_token']
            table_id = save_to_feishu['feishu_config'][0]['table_id']
            record_ids = save_to_feishu['feishu_record_ids']

            if app_token is None or table_id is None or record_ids is None or len(record_ids) == 0:
                raise ValueError("app token, table id and record ids are required")

            FeishuManager(config).batch_update_records(app_token, table_id, results, record_ids, audio_field_name)
        else:
            pass

        return io.NodeOutput(audios, ui={"custom_audios": results})