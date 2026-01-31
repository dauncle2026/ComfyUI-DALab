# DA Flux2 Klein Node Usage
[English](flux2_klein.md) | [中文文档](flux2_klein_zh.md)

## 1. Basic Illustrations

### Basic T2I (Text-to-Image)
Simplest usage: After configuration, just enter prompts to generate images.

<img src="../assets/flux2_klein_single_t2i.jpg" width="80%" />

### Basic I2I (Image-to-Image)
Supports image-guided generation. Connect reference images to the `images` port for style transfer or repainting.

<img src="../assets/flux2_klein_single_i2i.jpg" width="80%" />

### Batch T2I: With Qwen LLM
Use **DA Qwen LLM** to generate creative prompts in batch for automated continuous generation.
[Qwen LLM Node Usage](../text/qwen_llm.md)

<img src="../assets/flux2_klein_multi_llm_t2i.jpg" width="80%" />

### Batch I2I: With Feishu
Use **DA Feishu Load** to read prompts and image URLs from a table for fully automated batch image editing.
[Feishu Node Usage](../tools/feishu.md)

<img src="../assets/flux2_feishu_table.jpg" width="80%" />
<img src="../assets/flux2_klein_multi_feishu_i2i.jpg" width="80%" />

## 2. Configuration Setup

**DA Flux2 Klein Config** node manages parameters for the Flux2 Klein model.
> Global Config: Use with [Global Config](../tools/global_config.md) node to manage runtime VRAM control.

<img src="../assets/flux2_klein_config.jpg" width="80%" />

| Parameter | Default | Description |
| :--- | :--- | :--- |
| text_encoder_model | Qwen 3.8B | Flux2 Klein dedicated Text Encoder. |
| vae_model | Flux2 VAE | Flux2 VAE model. |
| diffusion_model | Flux2 Klein | Flux2 Klein Diffusion Model (9B FP8). |
| steps | 4 | Sampling steps. Klein only needs 4 steps. |
| batch_size | 1 | Number of images per generation. |
| sampler | euler | Sampling algorithm. 'euler' recommended. |
| easycache | - | Enable model caching to significantly speed up continuous generation. |
| loras | - | Select LoRA models (supports stacking). |

**DA Flux2 Klein (Generation Node)**
Supports **images** list input, accepting up to 10 reference images for editing or guidance.

## 3. Environment Dependencies
**No special dependencies**. Just install **ComfyUI-DALab** extension to use.

## 4. Model Downloads
> **Note**: If you have already downloaded the models, you can use them directly.

#### 1. Diffusion Model (UNet)
Path: `models/diffusion_models/`

| Version | Description | Download |
| :--- | :--- | :--- |
| **Klein (9B FP8)** | Distilled & Quantized (9B), 4 steps, VRAM friendly | [Pending Update] |

#### 2. Text Encoder
Path: `models/text_encoders/`

| Version | Description | Download |
| :--- | :--- | :--- |
| **Qwen 3.8B (FP8)** | Flux2 Klein dedicated Text Encoder | [Pending Update] |

#### 3. VAE Model
Path: `models/vae/`

| Version | Description | Download |
| :--- | :--- | :--- |
| **Default** | Official VAE Model (ae.safetensors) | [Download](https://huggingface.co/black-forest-labs/FLUX.2-dev/blob/main/ae.safetensors) |
