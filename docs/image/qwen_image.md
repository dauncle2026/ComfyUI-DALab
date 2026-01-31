# DA Qwen Image Node Usage
[English](qwen_image.md) | [中文文档](qwen_image_zh.md)

## 1. Basic Illustrations

### Basic T2I (Text-to-Image)
Simplest usage: After configuration, just enter prompts to generate images.

<img src="../assets/qwen_image_single_t2v.jpg" width="80%" />

### Batch T2I: With Qwen LLM
Use **DA Qwen LLM** to generate creative prompts in batch for automated continuous generation.
[Qwen LLM Node Usage](../text/qwen_llm.md)

<img src="../assets/qwen_image_multi_llm_t2v.jpg" width="80%" />

### Batch T2I: With Feishu
Use **DA Feishu Load** to read prompts from a table for fully automated batch generation.
[Feishu Node Usage](../tools/feishu.md)

<img src="../assets/flux1_feishu_table.jpg" width="80%" />
<img src="../assets/qwen_image_multi_feishu_t2v.jpg" width="80%" />

## 2. Configuration Setup

**DA Qwen Image Config** node manages parameters for the Qwen Image model.
> Global Config: Use with [Global Config](../tools/global_config.md) node to manage runtime VRAM control.

<img src="../assets/qwen_image_config.jpg" width="80%" />

| Parameter | Default | Description |
| :--- | :--- | :--- |
| text_encoder_model | Qwen 2.5 VL | Qwen VL Text Encoder (FP8). |
| vae_model | Qwen VAE | Qwen Image dedicated VAE model. |
| diffusion_model | Qwen Image | Qwen Image Core Diffusion Model (FP8). |
| steps | 5 | Sampling steps. Default is 5. |
| batch_size | 1 | Number of images per generation. |
| cfg | 1.0 | CFG Scale. Default is 1.0 (Lightning LoRA). |
| shift | 3.10 | Sampling shift parameter. Default is 3.10. |
| sampler | euler | Sampling algorithm. 'euler' recommended. |
| scheduler | simple | Noise scheduler. 'simple' recommended. |
| negative_prompt | (Default) | Built-in general negative prompts. |
| easycache | - | Enable model caching to significantly speed up continuous generation. |
| loras | - | Select LoRA models. |

## 3. Environment Dependencies
**No special dependencies**. Just install **ComfyUI-DALab** extension to use.

## 4. Model Downloads
> **Note**: If you have already downloaded the models, you can use them directly.

#### 1. Diffusion Model (UNet)
Path: `models/diffusion_models/`

| Version | Description | Download |
| :--- | :--- | :--- |
| **Qwen Image** | Qwen Image Diffusion Model | [Download](https://huggingface.co/Comfy-Org/Qwen-Image_ComfyUI/tree/main/split_files/diffusion_models) |

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
