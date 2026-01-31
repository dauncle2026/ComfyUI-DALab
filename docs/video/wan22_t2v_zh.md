# DA Wan 2.2 T2V 节点说明
[English](wan22_t2v.md) | [中文文档](wan22_t2v_zh.md)

## 1. 基本示意

### 基础文生视频 (Basic T2V)
最简单的使用方式：配置完成后，输入提示词即可生成视频。

<img src="../assets/wan22_single_t2v.jpg" width="80%" />

### 批量文生视频：搭配 Qwen LLM
利用 **DA Qwen LLM** 批量生成创意提示词，实现自动化连续生成。
[Qwen LLM 节点说明](../text/qwen_llm.md)

<img src="../assets/wan22_multi_llm_t2v.jpg" width="80%" />

### 批量文生视频：搭配 Feishu 多维表格
利用 **DA Feishu Load** 读取表格中的提示词，实现全自动化的批量视频生产。
[Feishu 节点说明](../tools/feishu.md)

<img src="../assets/flux1_feishu_table.jpg" width="80%" />
<img src="../assets/wan22_multi_feishu_t2v.jpg" width="80%" />

## 2. 节点配置说明

**DA Wan2.2 T2V Config** 节点用于管理 Wan 2.2 T2V 模型的参数配置。
> Global Config (全局配置): 搭配 [Global Config](../tools/global_config.md) 节点使用，用来管理运行时的显存控制。

<img src="../assets/wan22_config.jpg" width="80%" />

| 参数名 | 默认值 | 说明 |
| :--- | :--- | :--- |
| text_encoder_model | UMT5 XXL | Wan T5 文本编码器 (FP8)。 |
| vae_model | Wan 2.1 VAE | Wan VAE 模型。 |
| high_model | Wan 2.2 High Noise | Wan 2.2 高噪点扩散模型 (14B)。 |
| low_model | Wan 2.2 Low Noise | Wan 2.2 低噪点扩散模型 (14B)。 |
| steps | 4 | 采样步数。默认为 4 步 (加速版)。 |
| batch_size | 1 | 单次生成的视频数量。 |
| cfg | 1.0 | 提示词引导系数。默认为 1.0。 |
| shift | 5.0 | 采样偏移参数。默认为 5.0。 |
| sampler | euler | 采样算法。推荐使用 euler。 |
| scheduler | simple | 噪声调度器。推荐使用 simple。 |
| negative_prompt | (默认负面词) | 负面提示词，节点内置了通用负面词。 |
| high_loras | - | 为 High Model 选择加载 LoRA 模型。 |
| low_loras | - | 为 Low Model 选择加载 LoRA 模型。 |

**DA Wan2.2 T2V (生成节点)**
支持 **width** (宽), **height** (高), **frame_count** (帧数), **fps** (帧率), **prompts** (提示词) 和 **seed** (种子) 输入。

| 参数名 | 默认值 | 说明 |
| :--- | :--- | :--- |
| width | 640 | 输出视频宽度。 |
| height | 360 | 输出视频高度。 |
| frame_count | 81 | 生成帧数。默认 81 帧 (约 5 秒 @ 16fps)。 |
| fps | 16.0 | 视频帧率。默认 16.0。 |

## 3. 环境依赖
**无特殊依赖**。安装 **ComfyUI-DALab** 插件即可直接使用。

## 4. 模型下载
> **提示**：如果您之前已经下载过相关模型，直接使用即可。

#### 1. Diffusion 模型
存放路径: `models/diffusion_models/`

| 模型版本 | 说明 | 下载地址 |
| :--- | :--- | :--- |
| **High Noise 14B** | Wan 2.2 High Noise 模型 (14B) | [待更新] |
| **Low Noise 14B** | Wan 2.2 Low Noise 模型 (14B) | [待更新] |

#### 2. Text Encoder (UMT5)
存放路径: `models/text_encoders/`

| 模型版本 | 说明 | 下载地址 |
| :--- | :--- | :--- |
| **UMT5 XXL** | Wan T5 文本编码器 | [待更新] |

#### 3. VAE 模型
存放路径: `models/vae/`

| 模型版本 | 说明 | 下载地址 |
| :--- | :--- | :--- |
| **Wan 2.1 VAE** | Wan VAE 模型 | [待更新] |
