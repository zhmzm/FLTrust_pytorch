# FLTrust_pytorch
Some codes refers to https://github.com/WHDY/FedAvg  
paper FLTrust from https://arxiv.org/pdf/2012.13995.pdf

The central dataset(100 samples) is random selected from test dataset.

Run the code

```asp
python FLTrustServer.py -nc 100 -cf 0.1 -E 5 -B 10 -mn mnist_2nn  -ncomm 1000 -iid 0 -lr 0.01 -vf 20 -g 0
``
