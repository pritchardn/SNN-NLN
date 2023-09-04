# SNN-NLN
An SNN implementation of the NLN architecture for RFI detection.


[![DOI](https://zenodo.org/badge/633437547.svg)](https://zenodo.org/badge/latestdoi/633437547)


Contains:
 - A PyTorch re-implementation of [this work](https://github.com/mesarcik/RFI-NLN) with 
updated auto-encoder architecture
 - Code to convert trained ANN models to SNNs using 
[SpikingJelly](https://pypi.org/project/spikingjelly/)
 - Implementation of the Spiking NLN (SNLN)

All code files are in `src/`. Replicating all results can be achieved by running replicate.py

## Installation
```
conda create -n snn-nln python=3.10
conda activate snn-nln
pip install -r src/requirements.txt
```
You may need extra instructions for installing PyTorch with Cuda / Rocm support.

### Data Dependencies
The data used in this project is not included in this repository.
You will need to download the datasets from [zenodo](https://zenodo.org/record/6724065) and unzip
them into `/data`.

### Dependencies
AOFlagger is also required. Installation instructions can be found 
[here](https://aoflagger.readthedocs.io/en/latest/).

## Licensing
This code is licensed under the MIT License. See LICENSE for more details.
