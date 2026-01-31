# Flux.1 集成节点使用说明

## 1. 环境与模型准备

### 环境依赖
**无额外依赖**。
安装即可使用：确保您的 ComfyUI 已更新至最新版本，安装 **ComfyUI-DALab** 后即可直接使用本节点。

### 模型下载
请将对应的模型文件下载至 ComfyUI 的 `models` 目录下对应的文件夹中：

#### 1. Diffusion 模型 (UNet)
存放路径: `models/diffusion_models/`

| 模型版本 | 说明 | 下载地址 |
| :--- | :--- | :--- |
| **Dev (BF16)** | 官方原版，最高画质，显存要求高 | [下载](https://huggingface.co/black-forest-labs/FLUX.1-dev/blob/main/flux1-dev.safetensors) |
| **Dev (FP8)** | FP8 量化版，RTX 40 系列及以上显卡支持加速 | [下载](https://huggingface.co/black-forest-labs/FLUX.1-dev-FP8/blob/main/flux1-dev-fp8.safetensors) |
| **Dev (NVFP4)** | NVFP4 量化版，RTX 50 系列及以上显卡支持加速 | [下载](https://huggingface.co/black-forest-labs/FLUX.1-dev-NVFP4/blob/main/flux1-dev-nvfp4.safetensors) |

#### 2. T5 文本编码器 (Text Encoder)
存放路径: `models/text_encoders/`

| 模型版本 | 说明 | 下载地址 |
| :--- | :--- | :--- |
| **(FP16)** | T5 FP16 原版，精度更高但显存占用大 | [下载](https://huggingface.co/comfyanonymous/flux_text_encoders/blob/main/t5xxl_fp16.safetensors) |
| **(FP8)** | T5 FP8 量化版 (Scaled)，节省显存 (推荐) | [下载](https://huggingface.co/comfyanonymous/flux_text_encoders/blob/main/t5xxl_fp8_e4m3fn_scaled.safetensors) |

#### 3. CLIP 文本编码器 (Text Encoder)
存放路径: `models/text_encoders/`

| 模型版本 | 说明 | 下载地址 |
| :--- | :--- | :--- |
| **Default** | CLIP ViT-L 模型，Flux 必须模型 | [下载](https://huggingface.co/comfyanonymous/flux_text_encoders/blob/main/clip_l.safetensors) |

#### 4. VAE 模型
存放路径: `models/vae/`

| 模型版本 | 说明 | 下载地址 |
| :--- | :--- | :--- |
| **Default** | Flux 官方 VAE 模型 | [下载](https://huggingface.co/black-forest-labs/FLUX.1-dev/blob/main/ae.safetensors) |

---

## 2. 基础使用

**DA Flux1** 采用 **“一次配置，重复使用”** 的设计理念，将复杂的模型加载与采样参数剥离。

1.  **初始化配置（仅需一次）**：
*   添加 **DA Flux1 Config** 节点。
*   选择对应的模型文件，调整采样参数（如步数、CFG 等）。
*   **运行一次** (Queue Prompt)。此时配置信息会自动保存到本地。
*   *注：配置完成后，您可以删除或禁用 Config 节点，直到需要修改参数为止。*

<img src="../assets/flux1_config.jpg" width="80%" />

2.  **日常生成**：
*   只需保留 **DA Flux1** 节点。
*   在 Prompts 输入框中输入提示词。
*   点击 **Queue Prompt** 即可生成。节点会自动读取之前保存的配置。

<img src="../assets/flux1_default.jpg" width="80%" />

## 3. 详细配置参数

使用 **DA Flux1 Config** 节点可以对模型及采样参数进行详细配置。配置完成后，这些设置将被自动保存，下次使用时无需重复配置。

> **Global Config (全局配置)**: 搭配 **[Global Config](../tools/global_config.md)** 节点使用，用来管理运行时的显存控制。

| 参数名 | 默认值 | 说明 |
| :--- | :--- | :--- |
| **clip_model** | clip_l.safetensors | CLIP 文本编码器，通常使用 clip_l。 |
| **t5_model** | t5xxl_fp8... | T5 文本编码器，推荐使用 FP8 版本以节省显存。 |
| **vae_model** | flux1-vae.safetensors | 图像解码 VAE 模型。 |
| **diffusion_model** | flux1-dev-fp8.safetensors | Flux 核心扩散模型 (Transformer)。 |
| **steps** | 20 | 采样步数。Dev 模型推荐 20-30 步，Schnell 模型 4 步即可。 |
| **batch_size** | 1 | 单次生成的图片数量。 |
| **guidance** | 3.5 | 提示词引导系数。Flux Dev 推荐 3.5，Schnell 推荐 1.0 (或更低)。 |
| **sampler** | euler | 采样算法。推荐使用 `euler`。 |
| **scheduler** | simple | 噪声调度器。推荐使用 `simple`。 |
| **easycache** | - | 开启模型缓存，显著提升连续生成的响应速度。 |
| **loras** | - | 选择加载 LoRA 模型（支持多个 LoRA 叠加）。 |

## 4. 批量图片生成

DA Flux1 节点原生支持批量列表输入。只要将提示词列表传入 `prompts` 端口，节点就会自动按序生成所有图片。

### 方法一：使用 Qwen LLM 生成创意列表
利用 **DA Qwen LLM** 节点（或其他大语言模型节点）批量生成一组提示词，直接连接到 Flux 节点，即可实现“一个创意，无限裂变”。

**[Qwen LLM 使用说明](../text/qwen_llm.md)**

<img src="../assets/flux1_list_llm.jpg" width="80%" />

### 方法二：使用飞书多维表格 (Feishu)
利用 **DA Feishu Load** 节点读取云端多维表格中的提示词列，实现自动化的批量生产。

**[Feishu 飞书集成说明](../tools/feishu.md)**

1.  在飞书多维表格中准备好提示词。
2.  使用 **DA Feishu Load** 读取指定列。
3.  连接到 **DA Flux1** 节点进行生成。
4.  (可选) 还可以配合 **DA Feishu Save** 将生成的图片自动回传至飞书表格。

<img src="../assets/flux1_feishu_table.jpg" width="80%" />
<img src="../assets/flux1_list_feishu.jpg" width="80%" />
