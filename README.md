# ComfyUI-DALab

[![ComfyUI](https://img.shields.io/badge/ComfyUI-Extension-blue)](https://github.com/comfyanonymous/ComfyUI)
[![Python](https://img.shields.io/badge/Python-3.12%2B-green)](https://www.python.org/)

[English](README.md) | [中文文档](README_ZH.md)

**—— Da Uncle's ComfyUI Toolkit**

ComfyUI-DALab is a collection of ComfyUI nodes focused on improving production efficiency. Its core aim is to encapsulate complex underlying operations and tedious logic flows into concise and easy-to-use single nodes, significantly simplifying the wiring complexity of workflows and thereby improving ComfyUI production efficiency. In addition, this collection integrates support for cutting-edge AI generation models, covering image, video, and audio generation, along with practical productivity tools.

## Version History

*   **v0.0.3** (Jan 2026)
    *   **Initial Release**: First version launched, including 40+ high-efficiency encapsulated nodes.
    *   **Core Support**: Comprehensive support for Flux, Qwen, Z-Image image generation; Wan 2.2 video generation; and IndexTTS, CosyVoice, VoxCPM audio cloning.
    *   **Workflow Integration**: Built-in Feishu (Lark) connector for automated productivity workflows.

## Key Features

*   **Image Generation**: Deep integration of top text-to-image models like **Flux.1 / Flux.2**, **Qwen-VL**, **Z-Image**, supporting LoRA and advanced controls.
*   **Video Creation**: Supports the full **Wan2.2 (Wanxiang)** series (Text-to-Video, Image-to-Video, Video-to-Video, Image Animation), as well as **LTX-Video** and **InfiniteTalk** digital human generation.
*   **Audio Cloning**: Integrates high-definition speech synthesis and cloning models like **CosyVoice 3**, **VoxCPM 1.5**, **IndexTTS 2**.
*   **Vision Tools**: Provides visual analysis tools such as **Florence 2** (Image Captioning), **SAM 2** (Automatic Segmentation), and **DWPose** (Pose Detection).
*   **Productivity Integration**: Exclusive support for **Feishu (Lark)** integration, enabling reading prompts directly from Feishu multidimensional tables or uploading generation results to Feishu.

## Installation

### Method 1: Using ComfyUI Manager (Recommended)

1.  Open **ComfyUI Manager**.
2.  Type `ComfyUI-DALab` in the search box.
3.  Click **Install** and restart ComfyUI.

### Method 2: Manual Installation

```bash
cd ComfyUI/custom_nodes
git clone https://github.com/dauncle2026/ComfyUI-DALab.git

cd ComfyUI-DALab
pip install -r requirements.txt
```

## Node List
### 1. Image Generation

| Model | Description | Usage |
| :--- | :--- | :--- |
| **Flux.1** | Black Forest Labs flagship T2I model, supporting Dev/Schnell | [Usage](docs/image/flux1.md) |
| **Flux.2** | Upgraded Flux series version, providing stronger detail generation capabilities | [Usage](docs/image/flux2.md) |
| **Flux Klein** | Quantized/distilled optimized version of Flux model, usable with only 4GB VRAM | [Usage](docs/image/flux2_klein.md) |
| **Qwen Image** | Alibaba Tongyi Qwen T2I model, supporting multi-resolution generation | [Usage](docs/image/qwen_image.md) |
| **Qwen Edit** | Alibaba Tongyi Qwen image editing model, supporting repainting and modification | [Usage](docs/image/qwen_image_edit.md) |
| **ZImage** | Alibaba Tongyi Lab open-source 6B parameter efficient T2I model | [Usage](docs/image/zimage.md) |

### 2. Video Generation

| Model | Description | Usage |
| :--- | :--- | :--- |
| **Wan 2.2 T2V** | Alibaba Wanxiang 2.2 Text-to-Video model (14B), cinema-grade quality | [Usage](docs/video/wan22_t2v.md) |
| **Wan 2.2 I2V** | Alibaba Wanxiang 2.2 Image-to-Video model, precise image animation | [Usage](docs/video/wan22_i2v.md) |
| **Wan 2.2 S2V** | Alibaba Wanxiang 2.2 Video-to-Video, powerful video style transfer | [Usage](docs/video/wan22_s2v.md) |
| **Wan Animate** | Alibaba Wanxiang 2.2 video-driven character animation model, supporting pose and expression transfer | [Usage](docs/video/wan22_t2v.md) |
| **LTX Video** | Lightricks open-source high-efficiency DiT video generation model | [Usage](docs/video/ltx2.md) |
| **Infinite Talk** | MeiGen-AI open-source digital human speaking and lip-sync generation | [Usage](docs/video/infinite_talk.md) |

### 3. Audio Generation

| Model | Description | Usage |
| :--- | :--- | :--- |
| **CosyVoice 3** | Alibaba Tongyi Lab open-source multi-lingual Zero-shot voice cloning | [Usage](docs/audio/cosyvoice3.md) |
| **VoxCPM 1.5** | OpenBMB open-source Tokenizer-free context-aware speech synthesis | [Usage](docs/audio/voxcpm15.md) |
| **IndexTTS 2** | Industrial-grade efficient controllable Zero-shot speech synthesis system | [Usage](docs/audio/indextts2.md) |

### 4. Text Generation

| Model | Description | Usage |
| :--- | :--- | :--- |
| **Qwen VL/LLM** | Alibaba Qwen series large language model and visual understanding model | [Usage](docs/text/qwen_llm.md) |

### 5. Other Models

| Feature | Description | Usage |
| :--- | :--- | :--- |
| **Feishu Integration** | Deep integration with Feishu (Lark) multidimensional tables, realizing automated workflow loops | [Usage](docs/tools/feishu.md) |
| **Vision Tools** | Integrated Florence-2 (Captioning), SAM 2 (Segmentation), DWPose (Pose) | [Usage](docs/tools/vision_tools.md) |
| **File Utils** | Enhanced file management tools, supporting custom save paths and formats | [Usage](docs/tools/file_utils.md) |

## Acknowledgements

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
