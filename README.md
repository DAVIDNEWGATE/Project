# Project

Tensorflow implementations of Iterative Pruning, K-means Quantization, Dynamic Surgery Network, Incremental Network Quantization

Codes are implemented according to reference paper and codes.

Masks and modified weights are produced on Single Machine.

Re-training process are conduced on Google Cloud

All hyperparameters are set in config.py except DNS(to replicate its open-sourse code on Caffee)

All codes excepet Analysis&Experiments part are documented.

densenet.py: essential DenseNet modified from https://github.com/LaurentMazare/deep-models/tree/master/densenet

densenetfinaltest.py: inference of DenseNet

## Iterative Pruning

densenetfinalprune.py: produce masks & pruned weights

dnet_prune.ipynb: re-training of Iterative Pruning

## K-means Quantization

densenetfinalkmeans.py: produce masks & K-Qed weights

dnet_kmeans.ipynb: re-training of codebooks after K-means Quantization

## Dynamic Surgery Network

densenetfinalDNS.py: DNS

## Incremental Network Quantization

densenetfinalinq.py: produce masks & INQed weights

dnet_INQ.ipynb: re-training of INQ

## Combinations:

Pruning+INQ: INQPruning.ipynb

Pruning+K-Q: KmeansPruning.ipynb

## Analysis&Experiments:

dnet_dns_analysis.ipynb

dnet_dns_analysis2.ipynb

dnet_INQ-analysis.ipynb

dnet_prune_analysis.ipynb


## References:

https://github.com/gstaff/tfzip/tree/master/tfzip

https://github.com/garion9013/impl-pruning-TF

https://github.com/yiwenguo/Dynamic-Network-Surgery

https://arxiv.org/abs/1510.00149

https://arxiv.org/abs/1702.03044

https://arxiv.org/abs/1608.04493
