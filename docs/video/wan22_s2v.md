# DA Wan 2.2 S2V Node Usage
[English](wan22_s2v.md) | [中文文档](wan22_s2v_zh.md)

## 1. Basic Illustrations

### 1. Image + Audio to Video
Generate a lip-sync video from a single reference image and an audio file.

<img src="../assets/wan22_s2v_single.jpg" width="80%" />

### 2. Video + Audio to Video
Generate a lip-sync video using an existing video as reference.
- **Reference**: Uses the **Last Frame** of the input video as the reference image.
- **Motion**: Uses the video content for motion reference.

<img src="../assets/wan22_s2v_single_by_video.jpg" width="80%" />

### 3. With Control Video
Use a **DWPose** video to control the pose and motion of the generated character.

<img src="../assets/wan22_s2v_single_with_control_frame.jpg" width="80%" />

### 4. Batch S2V: With Feishu
Use **DA Feishu Load** to read prompts and settings for automated batch generation.

<img src="../assets/wan22_s2v_feishu_table.jpg" width="80%" />
<img src="../assets/wan22_s2v_multi_feishu.jpg" width="80%" />

## 2. Configuration Setup

**DA Wan2.2 S2V Config** node manages parameters for the Wan 2.2 S2V usage.
> Global Config: Use with [Global Config](../tools/global_config.md) node.

| Parameter | Default | Description |
| :--- | :--- | :--- |
| text_encoder_model | UMT5 XXL | Wan T5 Text Encoder (FP8). |
| vae_model | Wan 2.1 VAE | Wan VAE model. |
| diffusion_model | Wan 2.2 S2V 14B | Wan 2.2 S2V Diffusion Model (14B). |
| audio_encoder_model | Wav2Vec2 | Audio encoder for sound feature extraction. |
| steps | 4 | Sampling steps. Default is 4. |
| cfg | 1.0 | CFG Scale. Default is 1.0. |
| shift | 5.0 | Model sampling shift parameter. Default is 5.0. |
| chunk_frame_count | 77 | Frames per generation segment. |
| chunk_motion_frame_count| 73 | Motion frames per segment. |
| scheduler | simple | Noise scheduler. |
| sampler | uni_pc | Sampler algorithm. 'uni_pc' recommended for S2V. |
| negative_prompt | (Default) | Built-in general negative prompts. |

**DA Wan2.2 S2V (Generation Node)**
Key Parameters:
- **first_frames**: Reference Image or Video.
- **audios**: Input Audio.
- **control_videos**: Optional Control Video (e.g., DWPose).
- **max_frame_count**: Limit total frames (-1 for no limit).

| Parameter | Default | Description |
| :--- | :--- | :--- |
| width | 640 | Output video width. |
| height | 360 | Output video height. |
| fps | 16.0 | Frame rate. Default 16.0. |

## 3. Environment Dependencies
**No special dependencies**. Just install **ComfyUI-DALab** extension.

## 4. Model Downloads
> **Note**: If you have already downloaded the models, you can use them directly.

#### 1. Diffusion Model (S2V)
Path: `models/diffusion_models/`

| Version | Description | Download |
| :--- | :--- | :--- |
| **S2V 14B** | Wan 2.2 S2V Diffusion Model (14B) | [Download](https://huggingface.co/Comfy-Org/Wan_2.2_ComfyUI_Repackaged/tree/main/split_files/diffusion_models) |

#### 2. Audio Encoder
Path: `models/audio_encoders/`

| Version | Description | Download |
| :--- | :--- | :--- |
| **Wav2Vec2** | Wav2Vec2 Large English | [Download](https://huggingface.co/Comfy-Org/Wan_2.2_ComfyUI_Repackaged/tree/main/split_files/audio_encoders) |

#### 3. Text Encoder / VAE
(Same as T2V/I2V models)
- **Text Encoder**: `models/text_encoders/` [Download](https://huggingface.co/Comfy-Org/Wan_2.2_ComfyUI_Repackaged/tree/main/split_files/text_encoders)
- **VAE**: `models/vae/` [Download](https://huggingface.co/Comfy-Org/Wan_2.2_ComfyUI_Repackaged/tree/main/split_files/vae)
