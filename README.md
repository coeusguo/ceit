# Incorporating Convolution Designs into Visual Transformers


This repository is the official implementation of CeiT (Convolution-enhanced image Transformer). It builds from [Data-Efficient Vision Transformer](https://github.com/facebookresearch/deit) and [timm](https://github.com/rwightman/pytorch-image-models)

For details see [Incorporating Convolution Designs into Visual Transformers](https://arxiv.org/pdf/2103.11816.pdf) by Kun Yuan, Shaopeng Guo, Ziwei Liu, Aojun Zhou, Fengwei Yu and Wei Wu

If you use this code for a paper please cite:

```
@article{DBLP:journals/corr/abs-2103-11816,
  author    = {Kun Yuan and
               Shaopeng Guo and
               Ziwei Liu and
               Aojun Zhou and
               Fengwei Yu and
               Wei Wu},
  title     = {Incorporating Convolution Designs into Visual Transformers},
  journal   = {CoRR},
  volume    = {abs/2103.11816},
  year      = {2021},
  url       = {https://arxiv.org/abs/2103.11816},
  archivePrefix = {arXiv},
  eprint    = {2103.11816},
  timestamp = {Wed, 24 Mar 2021 15:50:40 +0100},
  biburl    = {https://dblp.org/rec/journals/corr/abs-2103-11816.bib},
  bibsource = {dblp computer science bibliography, https://dblp.org}
}
```

# Model Zoo

We provide baseline CeiT models pretrained on ImageNet 2012.

| model name | epoch | acc@1 | acc@5 | #params | url |
| --- | --- | --- | --- | --- | --- |
| ceit_tiny_patch16_224 | 300 | 76.4 | 93.4 | 6.4M | [TODO](TODO) |
| ceit_tiny_patch16_384 | 300 | 78.8| 94.7 | 6.4M| [TODO](TODO) |
| ceit_small_patch16_224 | 300 | 82.0 | 95.9 | 24.2M | [TODO](TODO) |
| ceit_small_patch16_384 | 300 | 83.3 | 96.5 | 24.2M | [TODO](TODO) |
| ceit_base_patch16_224 | 100 | 81.8 | - | - | [TODO](TODO) |

Before using it, make sure you have the pytorch-image-models package [`timm==0.3.2`](https://github.com/rwightman/pytorch-image-models) by [Ross Wightman](https://github.com/rwightman) installed. Note that our work relies of the augmentations proposed in this library.

# Usage

First, clone the repository locally:
```
git clone https://github.com/coeusguo/ceit.git
```
Then, install PyTorch 1.7.0+ and torchvision 0.8.1+ and [pytorch-image-models 0.3.2](https://github.com/rwightman/pytorch-image-models):

```
conda install -c pytorch pytorch torchvision
pip install timm==0.3.2
```

## Data preparation

Download and extract ImageNet train and val images from http://image-net.org/.
The directory structure is the standard layout for the torchvision [`datasets.ImageFolder`](https://pytorch.org/docs/stable/torchvision/datasets.html#imagefolder), and the training and validation data is expected to be in the `train/` folder and `val` folder respectively:

```
/path/to/imagenet/
  train/
    class1/
      img1.jpeg
    class2/
      img2.jpeg
  val/
    class1/
      img3.jpeg
    class/2
      img4.jpeg
```

## Evaluation
To evaluate a pre-trained CeiT model on ImageNet val with a single GPU run:
```
python main.py --eval --model <model name> --resume /path/to/checkpoint --data-path /path/to/imagenet
```

## Training
To train CeiT-Tiny and Deit-small on ImageNet on a single node with 4 gpus for 300 epochs run:

CeiT-tiny
```
python -m torch.distributed.launch --nproc_per_node=4 --use_env main.py --model ceit_tiny_patch16_224 --batch-size 256 --data-path /path/to/imagenet
```

CeiT-small
```
python -m torch.distributed.launch --nproc_per_node=4 --use_env main.py --model ceit_small_patch16_224 --batch-size 256 --data-path /path/to/imagenet
```

To train CeiT-Tiny and Deit-small on ImageNet on a single node with 8 gpus for 100 epochs run:
CeiT-base
```
python -m torch.distributed.launch --nproc_per_node=4 --use_env main.py --model ceit_base_patch16_224 --batch-size 256 --data-path /path/to/imagenet
```
