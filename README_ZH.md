# ComfyUI-DALab (Dauncle)

[![ComfyUI](https://img.shields.io/badge/ComfyUI-Extension-blue)](https://github.com/comfyanonymous/ComfyUI)
[![Python](https://img.shields.io/badge/Python-3.10%2B-green)](https://www.python.org/)

[English](README.md) | [中文文档](README_ZH.md)

ComfyUI-DALab 是一个功能强大的 ComfyUI 自定义节点集合，汇集了当前最前沿的 AI 生成模型支持，涵盖图像、视频、音频生成以及实用的生产力工具。

## ✨ 主要特性

*   **图像生成**: 深度集成 **Flux.1 / Flux.2**, **Qwen-VL**, **ZImage** 等顶尖文生图模型，支持 LoRA 和高级控制。
*   **视频创作**: 支持 **Wan2.2 (万相)** 全系列（文生视频、图生视频、视频转视频、动画生成），以及 **LTX-Video** 和 **InfiniteTalk** 数字人生成。
*   **音频克隆**: 集成 **CosyVoice 3**, **VoxCPM 1.5**, **IndexTTS 2** 等高清语音合成与克隆模型。
*   **视觉工具**: 提供 **Florence 2** 图像描述、**SAM 2** 自动分割、**DWPose** 姿态检测等视觉分析工具。
*   **生产力集成**: 独家支持 **飞书 (Feishu)** 集成，可直接从飞书多维表格读取提示词或将生成结果上传至飞书。

## 📦 安装指南

### 方法 1: 使用 ComfyUI Manager (推荐)
1.  打开 ComfyUI Manager。
2.  搜索 `ComfyUI-DALab` 或 `dauncle`。
3.  点击安装并重启 ComfyUI。

### 方法 2: 手动安装
```bash
cd ComfyUI/custom_nodes
git clone https://github.com/dauncle2026/ComfyUI-DALab.git
cd ComfyUI-DALab
pip install -r requirements.txt
```

## 🧩 节点列表

本项目包含 46 个自定义节点，分类如下：

### 🎨 图像生成 (Image)
*   **Flux 系列**: 支持 Flux.1, Flux.2 及针对性优化的 Klein 版本。
*   **Qwen Image**: 通义万相图像生成与编辑。
*   **ZImage**: 智谱 AI 图像生成模型支持。

### 🎬 视频生成 (Video)
*   **Wan 2.2**: 阿里万相 2.2 全家桶（T2V, I2V, S2V, Animate）。
*   **LTX Video**: 轻量级高效视频生成。
*   **Infinite Talk**: 能够让静态肖像说话的数字人视频生成。

### 🎵 音频 & 语音 (Audio)
*   **CosyVoice 3**: 高质量多语言语音克隆。
*   **VoxCPM 1.5**: 强大的语音合成模型。
*   **IndexTTS 2**: 另一种风格的语音生成选择。

### 🧠 LLM & 视觉模型
*   **Qwen VL / LLM**: 通义千问视觉理解与对话模型。

### 🛠 工具 & 辅助 (Tools)
*   **Feishu 集成**: 读取/写入飞书多维表格，实现自动化工作流。
*   **计算机视觉**: SAM2 (分割), Florence2 (反推), DWPose (姿态)。
*   **文件处理**: 增强的图像/视频/音频保存与拼接工具。

## 📂 示例工作流

项目目录 `example_workflows` 中提供了丰富的示例工作流文件（包含预览图与 JSON），涵盖了绝大多数核心功能，拖入 ComfyUI 即可使用。

## 🌍 多语言支持

已全面支持中文界面。请确保你的 ComfyUI 设置了中文语言环境（或安装了翻译插件），节点名称和说明将自动显示为中文。

## ⚠️ 注意事项

*   **模型下载**: 大部分节点需要预先下载对应的模型文件，请参考 ComfyUI 控制台的报错提示将模型放置在正确的目录下。
*   **API Key**: 使用 Qwen API 或飞书集成功能时，请在对应节点的配置中填入正确的 API Key。

---
*Created by dauncle2026*
