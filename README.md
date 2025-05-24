# Qwen2.5-Omni GPTQ 4-bit Audio Metadata Fork 🎶

**Default Branch:** `feature/low-vram-audio-metadata`

This repository is a fork of [QwenLM/Qwen2.5-Omni](https://github.com/QwenLM/Qwen2.5-Omni) (Apache-2.0), featuring enhanced audio metadata generation specifically optimized for preprocessing on low-VRAM devices. All other upstream features—GPTQ-Int4 quantization, low-VRAM inference pipelines, and multimodal APIs—are inherited without modification.

---

## 🌟 Key Points of This Fork

### 🧠 Inference Model

This fork relies on the **7B parameter, GPTQ 4-bit quantized** variant of Qwen2.5-Omni. We aim to utilize the power of the 7B model while making it accessible to users with low-VRAM hardware.

### 🎵 Audio Metadata Generation

The script `generate_audio_metadata.py` has been introduced, providing the following features:

* 🔍 **Recursive scanning** of an input directory for audio files.
* 📁 **Creation of individual JSON metadata files** for each audio file.
* 📌 Customizable system prompt, with a sensible default provided (see `generate_audio_metadata.py`). Use the `--system-prompt` option to specify your own.

### 💻 Low-VRAM Requirement Profiling

Based on preliminary tests with an RTX 4070 Ti SUPER (16 GB VRAM), typical VRAM usage averages approximately 8 GB, with occasional (initial) peaks of 12 GB. The performance profile is practical for most common scenarios, though devices with less VRAM may experience limitations.

*Ongoing optimization is aimed at reducing these VRAM peaks, with a target minimum requirement of 12 GB (**already achieved**) and an ideal goal of 8 GB. Contributions to further reduce VRAM usage are **highly encouraged**.*

### 🌍 Environment Specification

This fork is tested using the original repository's installation instructions. Any environment compatible with the original repository should remain compatible here. If any issues arise, please open an issue within this repository.

---

## 🚀 Quick Start Guide

Follow these steps to get started quickly:

### 1. Clone the Repository

```bash
git clone https://github.com/Skelp/Qwen2.5-Omni-GPTQ-4bit-audio-metadata.git
cd Qwen2.5-Omni-GPTQ-4bit-audio-metadata
```

### 2. Set Up Your Python Environment 🐍

Use `example_environment.txt` as a reference to install core dependencies. Python version **3.10.7** has been tested; other versions might work but aren't validated. Note: `flash_attn`, `qwen_omni_utils`, and `gptqmodel` are **mandatory** dependencies.

### 3. Generate Audio Metadata 🎙️

```bash
CUDA_VISIBLE_DEVICES=0 python3 ./generate_audio_metadata.py \
  '{your_audio_directory}' \
  --model-path Qwen/Qwen2.5-Omni-7B-GPTQ-Int4
```

#### Command Line Options:

* 📂 `--audio_dir`: Path to your audio files.
* 🧠 `--model-path`: Path to `Qwen2.5-Omni-7B-GPTQ-Int4` model directory. Similar to the original repository, if the directory doesn't exist or no model is found, one is downloaded based on the model-path via HuggingFace.
* 📝 `--system-prompt`: Specify the system prompt for metadata generation.
* 🔍 `--debug-vram`: Enables more detailed print-outs, displaying VRAM statistics at various points.

---

## 💬 System Prompt

The default system prompt is based on [training instructions](https://github.com/ace-step/ACE-Step/blob/main/TRAIN_INSTRUCTION.md) provided by [ace-step/ACE-Step](https://github.com/ace-step/ACE-Step/tree/main).

---

## 🤝 Contributing

We welcome contributions, suggestions, and issue reports via Pull Requests to the `feature/low-vram-audio-metadata` branch.

---

## 📜 License

Licensed under Apache-2.0. Please refer to [LICENSE](LICENSE) for complete details.
