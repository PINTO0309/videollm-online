## <img src="demo/assistant_rectangle.png" height="25"> VideoLLM-online: Online Video Large Language Model for Streaming Video

<a href="https://showlab.github.io/videollm-online/" target="_blank"><img alt="Homepage" src="https://img.shields.io/badge/🌍 Homepage-d35400?color=d35400" /></a>
<a href="https://huggingface.co/spaces/chenjoya/videollm-online" target="_blank"><img alt="Demo" src="https://img.shields.io/badge/🤗 Hugging Face Spaces-ffc107?color=ffc107" /></a>
<a href="https://arxiv.org/abs/2406.11816" target="_blank"><img alt="Paper" src="https://img.shields.io/badge/📄 Paper-28a745?color=28a745" /></a>
<a href="https://huggingface.co/chenjoya/videollm-online-8b-v1plus" target="_blank"><img alt="Checkpoint" src="https://img.shields.io/badge/🤗 Hugging Face Models-2980b9?color=2980b9" /></a>
<a href="https://huggingface.co/datasets/chenjoya/videollm-online-chat-ego4d-134k" target="_blank"><img alt="Data" src="https://img.shields.io/badge/🤗 Hugging Face Datasets-8e44ad?color=8e44ad" /></a>

## Updates

### 2025-09-03
I made some updates to make sure this repo can be inference with the latest transformers==4.55.4 and pytorch==2.7.1+cu128. Similar versions should be okay. If you want to obtain a strong VLM-online model, I highly recommend you to finetune Qwen2.5VL-Instruct with the streaming EOS loss here. No need to use the weak text-only Llama3 again.

### 2025-04-24
Updates! Our new paper: [LiveCC: Learning Video LLM with Streaming Speech Transcription at Scale](https://huggingface.co/papers/2504.16030). All resources, including the training video data, have been released at [LiveCC Webpage](https://showlab.github.io/livecc/)

### TLDR

The first streaming video LLM, high speed (5 ~ 10 FPS on NVIDIA 3090 GPU, 10 ~ 15 FPS on A100GPU) on long-form videos (10 minutes), with SOTA performance on online/offline settings.

[![Click to Play](webpage/static/images/firstframe.png)](https://www.youtube.com/watch?v=09OoQwxQBoc)

### Introduction

This is the official implementation of [VideoLLM-online: Online Video Large Language Model for Streaming Video](https://arxiv.org/abs/2406.11816), CVPR 2024. Our paper introduces several interesting stuffs compared to popular image/video/multimodal models:

- **Online Video Streaming**: Unlike previous models that serve as offline mode (querying/responding to a full video), our model supports **online interaction within a video stream**. It can **proactively** update responses during a stream, such as recording activity changes or helping with the next steps in real time. Even GPT-4o, which is audio-driven, requires user voice interaction with the visual scene, not actual video streaming.

- **Cheap and Scalable Streaming Data Synthesis**: Current video datasets for training multimodal LLMs are mostly offline and unsuitable for training an online video language model. Our method **transforms any offline annotation into streaming dialogue data** by prompting open-source LLM. The model is entirely trained on Llama synthesized data.

- **Parallelized Real-Time Inference**: Our inference method **parallelizes** video encoding, LLM forwarding for video frames, and LLM response generation, arranging them asynchronously. This significantly enhances real-time performance, achieving 10-15 FPS on an A100 GPU.

### Quick Start

- (Recommended) Launch the gradio demo locally with:
```sh
python -m demo.app --resume_from_checkpoint chenjoya/videollm-online-8b-v1plus
```

But if there are some bugs with flash-attn, try to use

```
python -m demo.app --resume_from_checkpoint chenjoya/videollm-online-8b-v1plus --attn_implementation sdpa
```

- (Recommended) Launch the CLI locally with:
```sh
python -m demo.cli --resume_from_checkpoint chenjoya/videollm-online-8b-v1plus
```

- (Deprecated, HF Spaces too slow) Try demo at <a href="https://huggingface.co/spaces/chenjoya/videollm-online" target="_blank"> <img alt="Demo" src="https://img.shields.io/badge/🤗 Hugging Face Spaces-ffc107?color=ffc107" /> </a>

By passing ```--resume_from_checkpoint chenjoya/videollm-online-8b-v1plus```, the PEFT checkpoint will be automatically downloaded and applied to ```meta-llama/Meta-Llama-3-8B-Instruct```.

### Installation

Ensure you have Miniconda and Python version >= 3.10 installed, then run:

- ffmpeg 7.x must be installed first
  - PyTorch source will make ffmpeg installed, but it is an old version and usually make very low quality preprocessing. Please install newest ffmpeg following:

    ```bash
    wget https://johnvansickle.com/ffmpeg/releases/ffmpeg-release-amd64-static.tar.xz
    tar xvf ffmpeg-release-amd64-static.tar.xz
    rm ffmpeg-release-amd64-static.tar.xz
    mv ffmpeg-7.0.2-amd64-static ffmpeg
    ```
- Install dependencies

  ```bash
  # Fix CUDA PATH if necessary
  # Required if multiple versions of CUDA are installed
  # CUDA 13.x is not supported by any packages
  export CUDA_HOME=/usr/local/cuda-12.8
  export PATH=/usr/local/cuda-12.8/bin:$PATH
  export LD_LIBRARY_PATH=/usr/local/cuda-12.8/lib64:$LD_LIBRARY_PATH

  # Fix CUDA GPU Compute Capability to prevent unnecessary binary bloat
  # RTX3070 = 8.6
  # RTX6000 Ada = 8.9
  # https://developer.nvidia.com/cuda/gpus
  export TORCH_CUDA_ARCH_LIST="8.9"
  # Required to avoid out of memory
  # Even with 128GB RAM, the default settings cause OOM
  # By limiting the number of parallel builds to 2,
  # the RAM consumption limit can be kept to around 50GB.
  export MAX_JOBS=2

  uv sync

  git clone git+https://github.com/2noise/ChatTTS
  mv ChatTTS demo/rendering/
  ```

### Training and Evaluation

- Download streaming dialogue data from <a href="https://huggingface.co/datasets/chenjoya/videollm-online-chat-ego4d-134k" target="_blank"><img alt="Data" src="https://img.shields.io/badge/🤗 Hugging Face Datasets-8e44ad?color=8e44ad" /></a>

- Distributed preprocess video frames: 2 FPS and 384 resolution, then using ```google/siglip-large-patch16-384``` to extract CLS with avg pooled  3x3 spatial tokens. Please refer to instructions under [data/preprocess/](data/preprocess/).

- Refer to the examples under [scripts/](scripts/)

- If you are interested in our streaming dialogue data generation, please see [data/livechat/](https://github.com/showlab/videollm-online/tree/main/data/livechat)

### Model Zoo

#### [VideoLLM-online-8B-v1+](https://huggingface.co/chenjoya/videollm-online-8b-v1plus)
* LLM: meta-llama/Meta-Llama-3-8B-Instruct
* Vision Strategy:
    * Frame Encoder: google/siglip-large-patch16-384
    * Frame Tokens: CLS token + 3x3 average pooled spatial tokens
    * Frame FPS: 2 for training, 2~10 for inference
    * Frame Resolution: max resolution 384, with zero-padding to keep aspect ratio
    * Video Length: 10 minutes
* Training Data: Ego4D Narration Stream 113K + Ego4D GoalStep Stream 21K

#### [VideoLLM-online-8B-v1](...)
* LLM: meta-llama/Meta-Llama-3-8B-Instruct
* Vision Strategy:
    * Frame Encoder: google/siglip-large-patch16-384
    * Frame Tokens: CLS token
    * Frame FPS: 2 for training, 2~10 for inference
    * Frame Resolution: max resolution 384, with zero-padding to keep aspect ratio
    * Video Length: 60 minutes
* Training Data: Ego4D Narration Stream 113K + Ego4D GoalStep Stream 21K

### VideoLLM-online beyond Llama

This codebase has a very simple and clean implementation. You only need to change the inherited class from Llama to Mistral to achieve the Mistral version of VideoLLM-online. Please refer to the examples in [models/live_llama](models/live_llama/).

### Citation

```
@inproceedings{videollm-online,
  author       = {Joya Chen and Zhaoyang Lv and Shiwei Wu and Kevin Qinghong Lin and Chenan Song and Difei Gao and Jia-Wei Liu and Ziteng Gao and Dongxing Mao and Mike Zheng Shou},
  title        = {VideoLLM-online: Online Video Large Language Model for Streaming Video},
  booktitle    = {CVPR},
  year         = {2024},
}
```
