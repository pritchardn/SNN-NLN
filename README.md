# SNN-NLN
An SNN implementation of the NLN architecture for RFI detection.
Based on [this work](https://github.com/mesarcik/RFI-NLN).

The files in src/learning include a variety of scripts to play with tooling speicifcally.

## Getting Cuda in Shape
``sudo update-alternatives --display cuda
sudo update-alternatives --config cuda``

## The installation steps I took
- ``conda install pytorch torchvision torchaudio pytorch-cuda=11.7 -c pytorch -c nvidia``
- 