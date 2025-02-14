# DistillAV

Codes for our paper of [Audio-Visual Representation Learning via Knowledge Distillation from Speech Foundation Models](https://arxiv.org/abs/2502.05766).

## Usage

### Environment

Please refer to the exported [environment.yml](https://github.com/jxzhanggg/DistillAV/blob/main/environment.yml) file.

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

  Pretrained Dataset | Model Size | Google Drive |
| ----------- | ----------- | ----------- |
| LRS3      | Base      | [lrs_base_pt.pt](https://drive.google.com/file/d/1iP5csxtLYy_hoIvG2Pbv4crNssLGfIJE/view?usp=sharing) |
| LRS3   | Large       | [lrs_large_pt.pt](https://drive.google.com/file/d/1wOuE065wBsBUIUCDu9mMFlFboRRZAB3a/view?usp=sharing) |
| LRS3+Voxceleb2 (En) | Base | [vox_base_pt.pt](https://drive.google.com/file/d/1mimTgiVDklulf6gNsX3-dye0jcCQTHSY/view?usp=sharing) |
| LRS3+Voxceleb2 (En) | Large | [vox_large_pt.pt](https://drive.google.com/file/d/1Luvni8GsVoAugQHxJzg8clSa0WBawGcG/view?usp=sharing) |


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
