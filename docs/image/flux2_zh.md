# DA Flux2 节点说明
[English](flux2.md) | [中文文档](flux2_zh.md)

## 1. 基本示意

### 基础文生图 (Basic T2I)
最简单的使用方式：配置完成后，输入提示词即可生成图片。

<img src="../assets/flux2_single_t2i.jpg" width="80%" />

### 基础编辑图 (Basic I2I)
支持图像引导生成 (Img2Img)，将参考图像连接至 `images` 端口，实现风格迁移或重绘。

<img src="../assets/flux2_single_i2i.jpg" width="80%" />

### 批量文生图：搭配 Qwen LLM
利用 **DA Qwen LLM** 批量生成创意提示词，实现自动化连续生成。
[Qwen LLM 节点说明](../text/qwen_llm.md)

<img src="../assets/flux2_multi_llm_t2i.jpg" width="80%" />

### 批量编辑生图：搭配 Qwen LLM
结合 Qwen LLM 的批量提示词与参考图像，实现批量化的风格编辑。

<img src="../assets/flux2_multi_llm_i2i.jpg" width="80%" />

### 批量编辑图片：搭配 Feishu 多维表格
利用 **DA Feishu Load** 读取表格中的提示词与图像 URL，实现全自动化的批量图像编辑流。
[Feishu 节点说明](../tools/feishu.md)

<img src="../assets/flux2_feishu_table.jpg" width="80%" />
<img src="../assets/flux2_multi_feishu_i2i.jpg" width="80%" />

## 2. 节点配置说明

**DA Flux2 Config** 节点用于管理 Flux.2 模型的参数配置。
> Global Config (全局配置): 搭配 [Global Config](../tools/global_config.md) 节点使用，用来管理运行时的显存控制。

<img src="../assets/flux2_config.jpg" width="80%" />

| 参数名 | 默认值 | 说明 |
| :--- | :--- | :--- |
| text_encoder_model | Mistral Small | Flux2 文本编码器 (Mistral)。 |
| vae_model | Flux2 VAE | Flux2 专用 VAE 模型。 |
| diffusion_model | Flux2 | Flux2 核心扩散模型 (NVFP4 量化版)。 |
| steps | 20 | 采样步数。默认为 20。 |
| batch_size | 1 | 单次生成的图片数量。 |
| guidance | 4.0 | 提示词引导系数。Flux2 默认为 4.0。 |
| sampler | euler | 采样算法。推荐使用 euler。 |
| easycache | - | 开启模型缓存，显著提升连续生成的响应速度。 |
| loras | - | 选择加载 LoRA 模型（支持多个 LoRA 叠加）。 |

**DA Flux2 (生成节点)**
除了标准的宽高与提示词输入外，Flux.2 节点支持 **images** 列表输入，最多支持 10 张参考图像用于编辑或引导。

## 3. 环境依赖
**无特殊依赖**。安装 **ComfyUI-DALab** 插件即可直接使用。

## 4. 模型下载
> **提示**：如果您之前已经下载过 FLUX.2 相关模型，直接使用即可，无需重复下载。

若未下载，请参考下方列表放置于 ComfyUI 对应目录：

#### 1. Diffusion 模型 (UNet)
存放路径: `models/diffusion_models/`

| 模型版本 | 说明 | 下载地址 |
| :--- | :--- | :--- |
| **Dev (BF16)** | 官方原版，最高画质 | [下载](https://huggingface.co/black-forest-labs/FLUX.2-dev/blob/main/flux2-dev.safetensors) |
| **Dev (FP8)** | FP8 混合量化版 (Comfy-Org)，推荐大多数显卡使用 | [下载](https://huggingface.co/Comfy-Org/flux2-dev/blob/main/split_files/diffusion_models/flux2_dev_fp8mixed.safetensors) |
| **Dev (NVFP4)** | NVFP4 量化版，RTX 50 系列及以上显卡支持加速 | [下载](https://huggingface.co/black-forest-labs/FLUX.2-dev-NVFP4/blob/main/flux2-dev-nvfp4-mixed.safetensors) |


#### 2. Text Encoder (Mistral)
存放路径: `models/text_encoders/`

| 模型版本 | 说明 | 下载地址 |
| :--- | :--- | :--- |
| **Mistral Small (BF16)** | 官方原版，精度最高 | [下载](https://huggingface.co/Comfy-Org/flux2-dev/blob/main/split_files/text_encoders/mistral_3_small_flux2_bf16.safetensors) |
| **Mistral Small (FP8)** | FP8 量化版，推荐大多数显卡使用 | [下载](https://huggingface.co/Comfy-Org/flux2-dev/blob/main/split_files/text_encoders/mistral_3_small_flux2_fp8.safetensors) |
| **Mistral Small (NVFP4)** | NVFP4 (FP4 Mixed) 量化版，RTX 50+ 加速支持 | [下载](https://huggingface.co/Comfy-Org/flux2-dev/blob/main/split_files/text_encoders/mistral_3_small_flux2_fp4_mixed.safetensors) |

#### 3. VAE 模型
存放路径: `models/vae/`

| 模型版本 | 说明 | 下载地址 |
| :--- | :--- | :--- |
| **Default** | 官方 VAE 模型 (ae.safetensors) | [下载](https://huggingface.co/black-forest-labs/FLUX.2-dev/blob/main/ae.safetensors) |

#### 5. LoRA 模型 (可选)
存放路径: `models/loras/`

| 模型版本 | 说明 | 下载地址 |
| :--- | :--- | :--- |
| **8-Step Turbo** | 8步加速 LoRA，大幅提升生成速度 | [下载](https://huggingface.co/Comfy-Org/flux2-dev/blob/main/split_files/loras/Flux_2-Turbo-LoRA_comfyui.safetensors) |
