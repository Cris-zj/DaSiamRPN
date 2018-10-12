# DaSiamRPN

This repository includes PyTorch code for reproducing the results on VOT2018.

[**Distractor-aware Siamese Networks for Visual Object Tracking**](https://arxiv.org/pdf/1808.06048.pdf)  

Zheng Zhu<sup>\*</sup>, Qiang Wang<sup>\*</sup>, Bo Li<sup>\*</sup>, Wei Wu, Junjie Yan, and Weiming Hu 

*European Conference on Computer Vision (ECCV), 2018*


## Pretrained model for SiamRPN

- download the pretrained model from google drive: [SiamRPNBIG.model](https://drive.google.com/file/d/1-vNVZxfbIplXHrqMHiJJYWXYWsOIvGsf/view?usp=sharing), and put it in './code'.



## Install the prerequisites

- install pytorch, numpy, opencv following the instructions in the `run_install.sh`. Please do **not** use conda to install.

## Modify the test.py

- modify the **imagedir** and **gtdir** in test.py


# License
Licensed under an MIT license.


## Citing DaSiamRPN

If you find **DaSiamRPN** and **SiamRPN** useful in your research, please consider citing:

```
@inproceedings{Zhu_2018_ECCV,
  title={Distractor-aware Siamese Networks for Visual Object Tracking},
  author={Zhu, Zheng and Wang, Qiang and Bo, Li and Wu, Wei and Yan, Junjie and Hu, Weiming},
  booktitle={European Conference on Computer Vision},
  year={2018}
}

@InProceedings{Li_2018_CVPR,
  title = {High Performance Visual Tracking With Siamese Region Proposal Network},
  author = {Li, Bo and Yan, Junjie and Wu, Wei and Zhu, Zheng and Hu, Xiaolin},
  booktitle = {The IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
  year = {2018}
}
```
