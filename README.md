This is the pytorch code for the paper: [Extending and Analyzing Self-Supervising Learning Across Domains](https://arxiv.org/pdf/2004.11992.pdf).

It contains implementations for Supervision, Jigsaw, Rotation, Instance Discrimination and Autoencoding on 17 standardized 64x64 image datasets (MINC has been added since the writing of the original paper).

This repository originally started as a fork of https://github.com/srebuffi/residual_adapters, but grew to focus on SSRL instead.

Datasets can be downloaded here: https://drive.google.com/file/d/1og7w_E3_CQh_S0kuIEMIlfd7KOEl7_SB/view?usp=sharing
The above download contains train-val-test splits of 17 datasets, along with train/val splits containing only 10% of the images or using only half of the available classes (relating to Figures 4 and 7 in our paper).


If you find this repository or associated paper useful in your research, please cite:
```
@misc{wallace2020extending,
    title={Extending and Analyzing Self-Supervised Learning Across Domains},
    author={Bram Wallace and Bharath Hariharan},
    year={2020},
    eprint={2004.11992},
    archivePrefix={arXiv},
    primaryClass={cs.CV}
}
```

TODO:
* Currently saving entire models instead of state dict for legacy reasons (note this currently means you must load model in same directory as models.py)
* Add models
