# FLTrust_pytorch 
**Unofficial implementation** for FLTrust, if there is any problem, please let me know.  

paper FLTrust from https://arxiv.org/pdf/2012.13995.pdf

official implementation from https://arxiv.org/abs/2012.13995

Some codes refer to https://github.com/WHDY/FedAvg  

This code is not suitable for resnet, because of BN layers. [Here](https://github.com/zhmzm/FLAME) contains another version of fltrust and it can support models with BN layers.

2022-12-29:

Now it can support BN layers in ResNet and VGG. Please report bugs you encounter in the issue. I will fix it soon.

The central dataset(100 samples) is randomly selected from the test dataset.

# Backdoor in FL

**Our recent paper "Backdoor Federated Learning by Poisoning Backdoor-critical Layers" has been accepted in ICLR'24, please refer to the [Github repo](https://github.com/zhmzm/Poisoning_Backdoor-critical_Layers_Attack).**

# Quick Start

```asp
python FLTrustServer.py -nc 100 -cf 0.1 -E 5 -B 10 -mn mnist_2nn  -ncomm 1000 -iid 0 -lr 0.01 -vf 20 -g 0
```

