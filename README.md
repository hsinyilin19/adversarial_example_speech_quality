# Adversarial Example Speech Quality

Codes for the paper "On the robustness of non-intrusive speech quality model by adversarial examples".

This repository hosts the Pytorch codes for paper [On the robustness of non-intrusive speech quality model by adversarial examples (ICASSP 2023)](https://arxiv.org/abs/2211.06508) by Hsin-Yi Lin, Huan-Hsin Tseng, and Yu Tsao.

This work shows that deep speech quality predictors can be vulnerable to adversarial perturbations, where the prediction can be changed drastically by unnoticeable perturbations. In addition to exposing the vulnerability of deep speech quality predictors, we further explore and confirm the viability of adversarial training for strengthening robustness of models.



## Datasets
###  - [Voice Bank corpus](https://datashare.ed.ac.uk/handle/10283/2791) (VCTK)


### - [TIMIT](https://catalog.ldc.upenn.edu/LDC93S1) Acoustic-Phonetic continuous speech corpus


### - [DNS challenge](https://github.com/microsoft/DNS-Challenge/) DNS challenge speech corpus


## Run
### Stage 1- Adversarial perturbations 
- change data paths, onnx model path,s ave model path, output path
- change score transform (in attack_modules.py)
- run stage1_attack.py

### Stage 2- Model enhancement by adversarial examples
- change data paths, onnx model path, perturbation paths, save model path
- run stage2_enhancement.py


## Prerequisites
- [Python 3.7](https://www.python.org/)
- [PyTorch 1.11](https://pytorch.org/)
- [librosa 0.9](https://librosa.org/doc/latest/index.html)
- [Tensorboardx 2.5](https://pypi.org/project/tensorboardX/)
- [scikit-learn 1.0](https://pypi.org/project/scikit-learn/)
- [tqdm 4.64](https://pypi.org/project/tqdm/)
- [numpy 1.21 ](https://pypi.org/project/numpy/)
- [torchaudio 0.11](https://pypi.org/project/torchaudio/)
- [scipy 1.6](https://pypi.org/project/scipy/)
- [audioread 2.1](https://pypi.org/project/audioread/)


## Hardware
- NVIDIA V100 (32 GB CUDA memory) and 4 CPUs.
