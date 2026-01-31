# DA Flux1 Node Usage
[English](flux1.md) | [中文文档](flux1_zh.md)

## 1. Basic Illustrations

### Basic Generation
Simplest usage: After configuration, just enter prompts to generate.

<img src="../assets/flux1_default.jpg" width="80%" />

### Batch Generation: With Qwen LLM
Use **DA Qwen LLM** node to generate creative prompts in batch for automated continuous generation.
[Qwen LLM Node Usage](../text/qwen_llm.md)

<img src="../assets/flux1_list_llm.jpg" width="80%" />

### Batch Generation: With Feishu Multidimensional Table
Use **DA Feishu Load** to read cloud table content for automated production workflows.
[Feishu Node Usage](../tools/feishu.md)

<img src="../assets/flux1_feishu_table.jpg" width="80%" />
<img src="../assets/flux1_list_feishu.jpg" width="80%" />


## 2. Configuration Setup

**DA Flux1 Config** node manages model paths and sampling parameters. Configure once, reuse forever.
> Global Config: Use with [Global Config](../tools/global_config.md) node to manage runtime VRAM control.

<img src="../assets/flux1_config.jpg" width="80%" />

| Parameter | Default | Description |
| :--- | :--- | :--- |
| clip_model | CLIP L | CLIP Text Encoder, usually clip_l. |
| t5_model | T5 (FP8) | T5 Text Encoder, FP8 version recommended to save VRAM. |
| vae_model | Flux VAE | Image Decoding VAE Model. |
| diffusion_model | Flux Dev (FP8) | Flux Core Diffusion Model (Transformer). |
| steps | 20 | Sampling steps. 20-30 for Dev, 4 for Schnell. |
| batch_size | 1 | Number of images per generation. |
| guidance | 3.5 | Guidance scale. 3.5 for Dev. |
| sampler | euler | Sampling algorithm. 'euler' recommended. |
| scheduler | simple | Noise scheduler. 'simple' recommended. |
| easycache | - | Enable model caching to significantly speed up continuous generation. |
| loras | - | Select LoRA models (supports stacking multiple LoRAs). |

## 3. Environment Dependencies
**No special dependencies**. Just install **ComfyUI-DALab** extension to use.

## 4. Model Downloads
> **Note**: If you have already downloaded FLUX.1 models, you can use them directly without re-downloading.

If not, please refer to the list below and place them in the corresponding ComfyUI directory:

#### 1. Diffusion Model (UNet)
Path: `models/diffusion_models/`

| Version | Description | Download |
| :--- | :--- | :--- |
| Dev (BF16) | Official Original, Best Quality, High VRAM | [Download](https://huggingface.co/black-forest-labs/FLUX.1-dev/blob/main/flux1-dev.safetensors) |
| Dev (FP8) | FP8 Quantized, Accelerated on RTX 40+ | [Download](https://huggingface.co/black-forest-labs/FLUX.1-dev-FP8/blob/main/flux1-dev-fp8.safetensors) |
| Dev (NVFP4) | NVFP4 Quantized, Accelerated on RTX 50+ | [Download](https://huggingface.co/black-forest-labs/FLUX.1-dev-NVFP4/blob/main/flux1-dev-nvfp4.safetensors) |

#### 2. T5 Text Encoder
Path: `models/text_encoders/`

| Version | Description | Download |
| :--- | :--- | :--- |
| (FP16) | T5 FP16 Original, High Precision, High VRAM | [Download](https://huggingface.co/comfyanonymous/flux_text_encoders/blob/main/t5xxl_fp16.safetensors) |
| (FP8) | T5 FP8 Scaled, Saves VRAM (Recommended) | [Download](https://huggingface.co/comfyanonymous/flux_text_encoders/blob/main/t5xxl_fp8_e4m3fn_scaled.safetensors) |

#### 3. CLIP Text Encoder
Path: `models/text_encoders/`

| Version | Description | Download |
| :--- | :--- | :--- |
| Default | CLIP ViT-L Model, Required for Flux | [Download](https://huggingface.co/comfyanonymous/flux_text_encoders/blob/main/clip_l.safetensors) |

#### 4. VAE Model
Path: `models/vae/`

| Version | Description | Download |
| :--- | :--- | :--- |
| Default | Flux Official VAE Model | [Download](https://huggingface.co/black-forest-labs/FLUX.1-dev/blob/main/ae.safetensors) |
