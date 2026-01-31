# ComfyUI-DALab

[![ComfyUI](https://img.shields.io/badge/ComfyUI-Extension-blue)](https://github.com/comfyanonymous/ComfyUI)
[![Python](https://img.shields.io/badge/Python-3.12%2B-green)](https://www.python.org/)

[English](README.md) | [中文文档](README_ZH.md)

ComfyUI-DALab 是一个专注于提升生产效率的 ComfyUI 节点合集。

通过将复杂的底层操作与繁琐的逻辑流程封装为简洁易用的单个节点，从而大幅简化工作流的复杂度。

## 版本记录

*   **v0.0.3** (2026年1月)：首个版本上线，包含 20+ 个高效率封装节点。

## 安装指南

### 方法 1: 使用 ComfyUI Manager (推荐)

1.  打开 **ComfyUI Manager**。
2.  在搜索框中输入 `ComfyUI-DALab`。
3.  点击 **Install** 按钮并重启 ComfyUI。

### 方法 2: 手动安装

```bash
cd ComfyUI/custom_nodes
git clone https://github.com/dauncle2026/ComfyUI-DALab.git

cd ComfyUI-DALab
pip install -r requirements.txt
```

## 节点列表
### 1. 图像生成

| 模型 | 说明 | 文档 |
| :--- | :--- | :--- |
| **Flux.1** | Black Forest Labs 旗舰级文生图模型，支持 Dev/Schnell | [使用说明](docs/image/flux1_zh.md) |
| **Flux.2** | Flux 系列升级版本，提供更强的细节生成能力 | [使用说明](docs/image/flux2_zh.md) |
| **Flux2 Klein** | Flux2 模型的量化/蒸馏优化版 | [使用说明](docs/image/flux2_klein_zh.md) |
| **Qwen Image** | Qwen Image 文生图模型，支持多分辨率生成 | [使用说明](docs/image/qwen_image_zh.md) |
| **Qwen Image Edit** | Qwen Image Edit 图像编辑模型，支持重绘与修改 | [使用说明](docs/image/qwen_image_edit_zh.md) |
| **ZImage** | 阿里通义实验室开源的 6B 参数高效文生图模型 | [使用说明](docs/image/zimage_zh.md) |

### 2. 视频生成

| 模型 | 说明 | 文档 |
| :--- | :--- | :--- |
| **Wan 2.2 T2V** | 阿里万相 2.2 文生视频模型 (14B)，电影级画质 | [使用说明](docs/video/wan22_t2v.md) |
| **Wan 2.2 I2V** | 阿里万相 2.2 图生视频模型，精准的图像动态化 | [使用说明](docs/video/wan22_i2v.md) |
| **Wan 2.2 S2V** | 阿里万相 2.2 视频转视频，强大的视频风格转换 | [使用说明](docs/video/wan22_s2v.md) |
| **Wan Animate** | 阿里万相 2.2 视频驱动角色动画模型，支持姿态与表情迁移 | [使用说明](docs/video/wan22_t2v.md) |
| **LTX Video** | Lightricks 开源的高效率 DiT 视频生成模型 | [使用说明](docs/video/ltx2.md) |
| **Infinite Talk** | MeiGen-AI 开源的数字人说话与口型同步生成 | [使用说明](docs/video/infinite_talk.md) |

### 3. 音频生成

| 模型 | 说明 | 文档 |
| :--- | :--- | :--- |
| **CosyVoice 3** | 阿里通义实验室开源的多语言 Zero-shot 语音克隆 | [使用说明](docs/audio/cosyvoice3.md) |
| **VoxCPM 1.5** | OpenBMB 开源的 Tokenizer-free 上下文感知语音合成 | [使用说明](docs/audio/voxcpm15.md) |
| **IndexTTS 2** | 工业级高效可控的 Zero-shot 语音合成系统 | [使用说明](docs/audio/indextts2.md) |

### 4. 文本生成

| 模型 | 说明 | 文档 |
| :--- | :--- | :--- |
| **Qwen VL/LLM** | 阿里 Qwen 系列大语言模型与视觉理解模型 | [使用说明](docs/text/qwen_llm.md) |

### 5. 其他模型

| 功能 | 说明 | 文档 |
| :--- | :--- | :--- |
| **Feishu 集成** | 飞书多维表格深度集成，实现自动化工作流闭环 | [使用说明](docs/tools/feishu.md) |
| **Vision Tools** | 集成 Florence-2 (反推)、SAM 2 (分割)、DWPose (姿态) | [使用说明](docs/tools/vision_tools.md) |
| **File Utils** | 增强型文件管理工具，支持自定义保存路径与格式 | [使用说明](docs/tools/file_utils.md) |

## 鸣谢

*   [ComfyUI](https://github.com/comfyanonymous/ComfyUI)
*   [Black Forest Labs](https://github.com/black-forest-labs/flux)
*   [Qwen](https://github.com/QwenLM/Qwen)
*   [Z-Image](https://github.com/Tongyi-MAI/Z-Image)
*   [Wan (Wan-Video)](https://github.com/Wan-Video/Wan2.2)
*   [LTX Video](https://github.com/Lightricks/LTX-Video)
*   [Infinite Talk](https://github.com/MeiGen-AI/InfiniteTalk)
*   [CosyVoice](https://github.com/FunAudioLLM/CosyVoice)
*   [VoxCPM](https://github.com/OpenBMB/VoxCPM)
*   [IndexTTS](https://github.com/index-tts/index-tts)
*   [Florence](https://huggingface.co/microsoft/Florence-2-large)
*   [SAM](https://github.com/facebookresearch/segment-anything-2)
*   [DWPose](https://github.com/IDEA-Research/DWPose)

