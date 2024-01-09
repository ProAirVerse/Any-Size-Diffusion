# ASD-upscale

## Introduction

This repository provides the official PyTorch implementation of [Any Size Diffusion (ASD)](https://arxiv.org/abs/2308.16582) - Fast Seamless Tiled Diffusion (FSTD) Upscaler. 
<img src="assets/Figure.png" height="350px"/>

The  implementation is based on [StableSR](https://github.com/IceClear/StableSR) . With our Fast Seamless Tiled Diffusion, the upscaling process is accelerated by at least 2x. Additionally, we have improved the tiled VAE implementation in the original repository, resulting in even faster upscaling for high-resolution images (e.g., 4K image, accelerated by at least 4x).

## Installation

We provide an installation shell for installation:

```bash
bash install.sh
```

For model checkpoints preparation, download the pretrained models from [HuggingFace]([Iceclear/StableSR at main](https://huggingface.co/Iceclear/StableSR/tree/main)) , put `stablesr_000117.ckpt` and `vqgan_cfw_00011.ckpt` at root directory. 

## Usage

To upscale your images, follow these steps:

1. Put your input images (in `.png` or `.jpg` format) in the specified [input folder].
2. Run the following command:

```bash
python launch.py [upscale factor] [input folder] [output folder] --gpus 0
```

If you have multiple images in the input folder, we recommend using multiple GPUs:

```bash
python launch.py [upscale factor] [input folder] [output folder] \
--gpus [list of available GPUs, e.g., 0 1 2 3]
```

Please note that the specific details may vary depending on your setup. Ensure that you have the necessary dependencies and configurations in place for successful execution.

## Citation

If you find this repository useful, please kindly consider citing the following paper:

```
@article{zheng2023any,
  title={Any-Size-Diffusion: Toward Efficient Text-Driven Synthesis for Any-Size HD Images},
  author={Zheng, Qingping and Guo, Yuanfan and Deng, Jiankang and Han, Jianhua and Li, Ying and Xu, Songcen and Xu, Hang},
  journal={arXiv preprint arXiv:2308.16582},
  year={2023}
}
```