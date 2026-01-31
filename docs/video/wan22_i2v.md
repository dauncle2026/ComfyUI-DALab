# DA Wan 2.2 I2V Node Usage
[English](wan22_i2v.md) | [中文文档](wan22_i2v_zh.md)

## 1. Basic Illustrations

### 1. First Frame to Video
Generate a video starting from a single image. Connect the image to `first_frames`.

<img src="../assets/wan22_i2v_first_single.jpg" width="80%" />

### 2. First & Last Frame to Video
Generate a video that transitions between two images. Connect images to `first_frames` and `last_frames`.

<img src="../assets/wan22_i2v_first_last_single.jpg" width="80%" />

### 3. Video Bridge (Video to Video)
Generate a transition video between two existing videos.
- **First Frame Input**: Accepts a video, automatically uses its **last frame**.
- **Last Frame Input**: Accepts a video, automatically uses its **first frame**.

<img src="../assets/wan22_i2v_first_last_single_by_video.jpg" width="80%" />

### 4. Batch I2V: With Qwen LLM
Use **DA Qwen LLM** to generate multiple prompts for batch video generation.

<img src="../assets/wan22_i2v_first_multi_llm.jpg" width="80%" />

## 2. Configuration Setup

**DA Wan2.2 I2V Config** node manages parameters for the Wan 2.2 I2V model.
> Global Config: Use with [Global Config](../tools/global_config.md) node to manage runtime VRAM control.

<img src="../assets/wan22_config.jpg" width="80%" />

| Parameter | Default | Description |
| :--- | :--- | :--- |
| text_encoder_model | UMT5 XXL | Wan T5 Text Encoder (FP8). |
| vae_model | Wan 2.1 VAE | Wan VAE model. |
| high_model | Wan 2.2 I2V High | Wan 2.2 I2V High Noise Diffusion Model (14B). |
| low_model | Wan 2.2 I2V Low | Wan 2.2 I2V Low Noise Diffusion Model (14B). |
| steps | 4 | Sampling steps. Default is 4 (Lightning). |
| batch_size | 1 | Number of videos per generation. |
| cfg | 1.0 | CFG Scale. Default is 1.0. |
| shift | 5.0 | Model sampling shift parameter. Default is 5.0. |
| sampler | euler | Sampling algorithm. |
| scheduler | simple | Noise scheduler. |
| negative_prompt | (Default) | Built-in general negative prompts. |
| high_loras | - | Select LoRA models for High Model. |
| low_loras | - | Select LoRA models for Low Model. |

**DA Wan2.2 I2V (Generation Node)**
Key Parameters:
- **first_frames**: Input for the start of the video. Supports **Image** or **Video**.
    - If **Video** is provided, the **Last Frame** of that video is used.
- **last_frames**: Input for the end of the video. Supports **Image** or **Video**.
    - If **Video** is provided, the **First Frame** of that video is used.

| Parameter | Default | Description |
| :--- | :--- | :--- |
| width | 640 | Output video width. |
| height | 360 | Output video height. |
| frame_count | 81 | Number of frames (duration). Default 81 frames. |
| fps | 16.0 | Frame rate. Default 16.0. |

## 3. Environment Dependencies
**No special dependencies**. Just install **ComfyUI-DALab** extension to use.

## 4. Model Downloads
> **Note**: If you have already downloaded the models, you can use them directly.

#### 1. Diffusion Models (I2V)
Path: `models/diffusion_models/`

| Version | Description | Download |
| :--- | :--- | :--- |
| **I2V High Noise 14B** | Wan 2.2 I2V High Noise Model (14B) | [Download](https://huggingface.co/Comfy-Org/Wan_2.2_ComfyUI_Repackaged/tree/main/split_files/diffusion_models) |
| **I2V Low Noise 14B** | Wan 2.2 I2V Low Noise Model (14B) | [Download](https://huggingface.co/Comfy-Org/Wan_2.2_ComfyUI_Repackaged/tree/main/split_files/diffusion_models) |

#### 2. Text Encoder (UMT5)
Path: `models/text_encoders/`

| Version | Description | Download |
| :--- | :--- | :--- |
| **UMT5 XXL** | Wan T5 Text Encoder | [Download](https://huggingface.co/Comfy-Org/Wan_2.2_ComfyUI_Repackaged/tree/main/split_files/text_encoders) |

#### 3. VAE Model
Path: `models/vae/`

| Version | Description | Download |
| :--- | :--- | :--- |
| **Wan 2.1 VAE** | Wan VAE Model | [Download](https://huggingface.co/Comfy-Org/Wan_2.2_ComfyUI_Repackaged/tree/main/split_files/vae) |

#### 4. LoRA Model (Optional)
Path: `models/loras/`

| Version | Description | Download |
| :--- | :--- | :--- |
| **Lightx2v** | Acceleration LoRA | [Download](https://huggingface.co/Comfy-Org/Wan_2.2_ComfyUI_Repackaged/tree/main/split_files/loras) |
