# DA Qwen Image Edit Node Usage
[English](qwen_image_edit.md) | [中文文档](qwen_image_edit_zh.md)

## 1. Basic Illustrations

### Basic Image Editing (Single I2I)
Supports image editing/repainting. Connect an input image to the `images` port and provide a prompt to guide the editing process.

<img src="../assets/qwen_image_edit_single_i2i.jpg" width="80%" />

### Batch Image Editing: With Feishu
Use **DA Feishu Load** to read prompts and image URLs from a table for fully automated batch image editing.
[Feishu Node Usage](../tools/feishu.md)

<img src="../assets/flux2_feishu_table.jpg" width="80%" />
<img src="../assets/qwen_image_edit_multi_i2i.jpg" width="80%" />

## 2. Configuration Setup

**DA Qwen Image Edit Config** node manages parameters for the Qwen Image Edit model.
> Global Config: Use with [Global Config](../tools/global_config.md) node to manage runtime VRAM control.

<img src="../assets/qwen_image_edit_config.jpg" width="80%" />

| Parameter | Default | Description |
| :--- | :--- | :--- |
| text_encoder_model | Qwen 2.5 VL | Qwen VL Text Encoder (FP8). |
| vae_model | Qwen VAE | Qwen dedicated VAE model. |
| diffusion_model | Qwen Image Edit | Qwen Image Edit Core Diffusion Model. |
| steps | 4 | Sampling steps. Default is 4. |
| batch_size | 1 | Number of images per generation. |
| cfg | 1.0 | CFG Scale. |
| shift | 3.10 | Model sampling shift parameter. Default is 3.10. |
| cfg_norm_strength | 1.0 | CFG Norm Strength. Default is 1.0. |
| sampler | euler | Sampling algorithm. |
| scheduler | simple | Noise scheduler. |
| negative_prompt | - | Negative prompt. Default is empty. |
| easycache | - | Enable model caching to significantly speed up continuous generation. |
| loras | - | Select LoRA models. |

**DA Qwen Image Edit (Generation Node)**
Requires both **prompts** and **images** (image) input.

## 3. Environment Dependencies
**No special dependencies**. Just install **ComfyUI-DALab** extension to use.

## 4. Model Downloads
> **Note**: If you have already downloaded the models, you can use them directly.

#### 1. Diffusion Model (UNet)
Path: `models/diffusion_models/`

| Version | Description | Download |
| :--- | :--- | :--- |
| **Qwen Image Edit** | Qwen Image Edit Diffusion Model | [Download](https://huggingface.co/Comfy-Org/Qwen-Image-Edit_ComfyUI/tree/main/split_files/diffusion_models) |

#### 2. Text Encoder (VL)
Path: `models/text_encoders/`

| Version | Description | Download |
| :--- | :--- | :--- |
| **Qwen 2.5 VL** | Qwen 2.5 VL Model | [Download](https://huggingface.co/Comfy-Org/Qwen-Image_ComfyUI/tree/main/split_files/text_encoders) |

#### 3. VAE Model
Path: `models/vae/`

| Version | Description | Download |
| :--- | :--- | :--- |
| **Qwen Image VAE** | Dedicated VAE Model | [Download](https://huggingface.co/Comfy-Org/Qwen-Image_ComfyUI/tree/main/split_files/vae) |

#### 4. LoRA Model (Optional)
Path: `models/loras/`

| Version | Description | Download |
| :--- | :--- | :--- |
| **Lightx2v** | Acceleration LoRA | [Download](https://huggingface.co/lightx2v/Qwen-Image-Lightning/tree/main) |
| **Multiple Angles** | Multi-view generation LoRA | [Download](https://huggingface.co/fal/Qwen-Image-Edit-2511-Multiple-Angles-LoRA) |
