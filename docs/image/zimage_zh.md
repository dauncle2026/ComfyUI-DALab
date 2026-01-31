# DA ZImage 节点说明
[English](zimage.md) | [中文文档](zimage_zh.md)

## 1. 基本示意

### 基础文生图 (Basic T2I)
最简单的使用方式：配置完成后，输入提示词即可生成图片。

<img src="../assets/zimage_single_t2v.jpg" width="80%" />

### 批量文生图：搭配 Qwen LLM
利用 **DA Qwen LLM** 批量生成创意提示词，实现自动化连续生成。
[Qwen LLM 节点说明](../text/qwen_llm.md)

<img src="../assets/zimage_multi_llm_t2v.jpg" width="80%" />

### 批量文生图：搭配 Feishu 多维表格
利用 **DA Feishu Load** 读取表格中的提示词，实现全自动化的批量生产。
[Feishu 节点说明](../tools/feishu.md)

<img src="../assets/zimage_multi_feishu_t2v.jpg" width="80%" />

## 2. 节点配置说明

**DA ZImage Config** 节点用于管理 ZImage 模型的参数配置。
> Global Config (全局配置): 搭配 [Global Config](../tools/global_config.md) 节点使用，用来管理运行时的显存控制。

<img src="../assets/zimage_config.jpg" width="80%" />

| 参数名 | 默认值 | 说明 |
| :--- | :--- | :--- |
| text_encoder_model | Qwen 3 4B | ZImage 文本编码器 (Qwen 3 4B)。 |
| vae_model | ZImage VAE | ZImage 专用 VAE 模型。 |
| diffusion_model | ZImage | ZImage 核心扩散模型。 |
| steps | 9 | 采样步数。默认为 9 步。 |
| batch_size | 1 | 单次生成的图片数量。 |
| cfg | 1.0 | 提示词引导系数。 |
| shift | 3.0 | 采样偏移参数。默认为 3.0。 |
| sampler | res_multistep | 采样算法。推荐使用 res_multistep。 |
| scheduler | simple | 噪声调度器。 |

| easycache | - | 开启模型缓存，显著提升连续生成的响应速度。 |
| loras | - | 选择加载 LoRA 模型。 |

**DA ZImage (生成节点)**
仅支持 **prompts** (提示词) 列表输入，用于批量文生图。

## 3. 环境依赖
**无特殊依赖**。安装 **ComfyUI-DALab** 插件即可直接使用。

## 4. 模型下载
> **提示**：如果您之前已经下载过相关模型，直接使用即可。

#### 1. Diffusion 模型 (UNet)
存放路径: `models/diffusion_models/`

| 模型版本 | 说明 | 下载地址 |
| :--- | :--- | :--- |
| **ZImage** | ZImage 扩散模型 | [待更新] |

#### 2. Text Encoder
存放路径: `models/text_encoders/`

| 模型版本 | 说明 | 下载地址 |
| :--- | :--- | :--- |
| **ZImage Text Encoder** | ZImage 文本编码器 | [待更新] |

#### 3. VAE 模型
存放路径: `models/vae/`

| 模型版本 | 说明 | 下载地址 |
| :--- | :--- | :--- |
| **ZImage VAE** | 专用 VAE 模型 | [待更新] |
