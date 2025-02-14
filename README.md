# DistillAV

Codes for our paper of [Audio-Visual Representation Learning via Knowledge Distillation from Speech Foundation Models](https://arxiv.org/abs/2502.05766).

## Usage

### Data preparation

Please refer to [AV-HuBERT](https://github.com/facebookresearch/av_hubert) code repo. Only difference
is that our models use entire faces as visual inputs, rather than the extracted lip-ROI.

### Pretraining

```bash
cd run_scripts
bash run_alignmtl.sh
```
### Finetune

```bash
cd run_scripts/run_finetune
bash run_finetune.sh
```

## Pretrained Models

Our pretrained models are coming soon.
Please note that our model accept full face images of LRS3/Voxceleb video as inputs.

## Citation

You can cite our paper by

```bibtex
@article{ZHANG2025111432,
title = {Audio-visual representation learning via knowledge distillation from speech foundation models},
journal = {Pattern Recognition},
volume = {162},
pages = {111432},
year = {2025},
issn = {0031-3203},
doi = {https://doi.org/10.1016/j.patcog.2025.111432},
url = {https://www.sciencedirect.com/science/article/pii/S0031320325000925},
author = {Jing-Xuan Zhang and Genshun Wan and Jianqing Gao and Zhen-Hua Ling},
}
```

## Acknowledgements

Our code was adapted from the following project:
* https://github.com/facebookresearch/av_hubert
