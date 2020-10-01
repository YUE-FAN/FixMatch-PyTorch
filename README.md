# FixMatch-PyTorch-Reproduction
This repo contains a PyTorch implementation of [FixMatch: Simplifying Semi-Supervised Learning with Consistency and Confidence](https://arxiv.org/abs/2001.07685).
The official Tensorflow implementation is [here](https://github.com/google-research/fixmatch).

This code reproduces CIFAR-10 and CIFAR-100 RA results in the paper.

## Usage
Train the model on CIFAR-10 with 40 labels:
```
python cifar_fixmatch_reproduce.py -a wideresnetleaky --k 2 --n 28 -d cifar10 -j 4 --epochs 1024 --train_batch 64 --lr 0.03 --init_data 4 --val_data 1 --mu 7 --lambda_u 1 --threshold 0.95 --n_imgs_per_epoch 65536 --checkpoint YOUR_PATH --manualSeed 1 --datasetSeed 1 --use_ema --ema_decay 0.999 --wd 0.0005 --gpu-id 0
```

Train the model on CIFAR-100 with 400 labels (the model used on CIFAR-100 is larger, see [here](https://github.com/google-research/fixmatch/issues/25)):
```
python cifar_fixmatch_reproduce.py -a wideresnetleaky --k 8 --n 28 -d cifar100 -j 4 --epochs 1024 --train_batch 64 --lr 0.03 --init_data 4 --val_data 1 --mu 7 --lambda_u 1 --threshold 0.95 --n_imgs_per_epoch 65536 --checkpoint YOUR_PATH --manualSeed 1 --datasetSeed 1 --use_ema --ema_decay 0.999 --wd 0.0005 --gpu-id 0
```

## Results
### CIFAR-10 error rate
| #Labels | 40 | 250 | 4000 |
|:---|:---:|:---:|:---:|
|Paper (RA) | 13.81 ± 3.37 | 5.07 ± 0.65 | 4.26 ± 0.05 |
|This code | 6.19 | 4.32 | 3.96 |

### CIFAR-100 error rate
| #Labels | 400 | 2500 | 10000 |
|:---|:---:|:---:|:---:|
|Paper (RA) | 48.85 ± 1.75 | 28.29 ± 0.11 | 22.60 ± 0.12 |
|This code | 44.18 | 26.37 | 21.37 |

\* Results of this code were evaluated on 1 run.

## References
- official implement https://github.com/google-research/fixmatch
- https://github.com/CoinCheung/fixmatch 
- https://github.com/kekmodel/FixMatch-pytorch
- https://github.com/valencebond/FixMatch_pytorch
