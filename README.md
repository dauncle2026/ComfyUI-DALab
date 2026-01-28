# ComfyUI-DALab (Dauncle)

[![ComfyUI](https://img.shields.io/badge/ComfyUI-Extension-blue)](https://github.com/comfyanonymous/ComfyUI)
[![Python](https://img.shields.io/badge/Python-3.10%2B-green)](https://www.python.org/)

[English](README.md) | [中文文档](README_ZH.md)

ComfyUI-DALab is a powerful collection of custom nodes for ComfyUI, aggregating cutting-edge AI generation models. It covers image, video, and audio generation, along with practical productivity tools.

## ✨ Key Features

*   **Image Generation**: Deep integration with top-tier T2I models like **Flux.1 / Flux.2**, **Qwen-VL**, and **ZImage**, featuring LoRA support and advanced control.
*   **Video Creation**: Full support for the **Wan2.2 (Wanxiang)** suite (T2V, I2V, S2V, Animation), as well as **LTX-Video** and **InfiniteTalk** for digital human generation.
*   **Audio Cloning**: Integrated high-quality TTS and voice cloning models including **CosyVoice 3**, **VoxCPM 1.5**, and **IndexTTS 2**.
*   **Visual Tools**: Provides visual analysis tools such as **Florence 2** for captioning, **SAM 2** for automated segmentation, and **DWPose** for pose detection.
*   **Productivity Integration**: Exclusive support for **Feishu (Lark)** integration, allowing prompt reading from Feishu Base or uploading generation results directly to Feishu.

## 📦 Installation

### Method 1: ComfyUI Manager (Recommended)
1.  Open ComfyUI Manager.
2.  Search for `ComfyUI-DALab` or `dauncle`.
3.  Click Install and restart ComfyUI.

### Method 2: Manual Installation
```bash
cd ComfyUI/custom_nodes
git clone https://github.com/dauncle2026/ComfyUI-DALab.git
cd ComfyUI-DALab
pip install -r requirements.txt
```

## 🧩 Node List

This project includes 46 custom nodes, categorized as follows:

### 🎨 Image Generation
*   **Flux Series**: Supports Flux.1, Flux.2, and the optimized Klein version.
*   **Qwen Image**: Tongyi Wanxiang image generation and editing.
*   **ZImage**: Support for Zhipu AI image generation models.

### 🎬 Video Generation
*   **Wan 2.2**: Ali Wanxiang 2.2 full suite (T2V, I2V, S2V, Animate).
*   **LTX Video**: Lightweight and efficient video generation.
*   **Infinite Talk**: Digital human video generation capable of animating static portraits.

### 🎵 Audio & Speech
*   **CosyVoice 3**: High-quality multilingual voice cloning.
*   **VoxCPM 1.5**: Powerful speech synthesis model.
*   **IndexTTS 2**: Another excellent option for speech generation.

### 🧠 LLM & Visual Models
*   **Qwen VL / LLM**: Tongyi Qianwen visual understanding and dialogue models.

### 🛠 Tools & Utilities
*   **Feishu Integration**: Read/Write Feishu Base tables for automated workflows.
*   **Computer Vision**: SAM2 (Segmentation), Florence2 (Captioning), DWPose (Pose Estimation).
*   **File Handling**: Enhanced tools for saving and concatenating images/videos/audio.

## 📂 Example Workflows

The `example_workflows` directory contains a rich set of example workflow files (including preview images and JSONs) covering most core features. Simply drag and drop them into ComfyUI to start using them.

## 🌍 Multilingual Support

The interface fully supports Chinese. If your ComfyUI is configured for Chinese (or has a translation plugin), node names and descriptions will automatically appear in Chinese.

## ⚠️ Notes

*   **Model Downloads**: Most nodes require pre-downloaded model files. Please refer to the error messages in the ComfyUI console to place models in the correct directories.
*   **API Keys**: When using Qwen API or Feishu integration features, please enter the correct API Keys in the corresponding node configurations.

---
*Created by dauncle2026*
