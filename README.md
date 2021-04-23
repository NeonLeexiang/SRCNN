# SRCNN--Using Tensorflow 2.0

作为一个简单的论文复现，用于了解 `Tensorflow` 同时也更加深入了解卷积神经网络。

## Prerequisites
 * Tensorflow  > 2.0  
也是想着通过这个项目去尝试使用 `Tensorflow 2.0` 然后复现一下超分辨比较经典的论文 `SRCNN` . 


## Usage
For training, `python trains.py`
<br>
For testing, `python trains.py`
但是需要注释一些内容


## Problems
* 因为使用的是 `cifar10`的数据集，会出现的问题就是它的图像数据的大小是 32*32 的，
  所以没有做一些放大缩小的操作获取对应的 High Resolution Image -> Low Resolution Image 的操作。
  
* 做的 `Keras` 和 `Tensorflow` 的训练并没有像 `Pytorch` 一样使用 `tqdm` 模块去做一些操作。  
  
* `pytorch` 要非常注意一点就是它的 Tensor 和 `tensorflow` 或者 `keras` 不一样，可能 `tensorflow` `keras` 是以
  `Size * H * W * C` 而 `pytorch` 是以 `Size * C * H * W` 的方式去计算的，所以使用的数据需要通过 `torch.permute` 的 方式修改数据格式。  
    
  

## Result
  
做一个 `result table` :  

| Dataset | Epochs | Module | Method     | psnr   |
|---------|------- |------  |------      | ------ |
| cifar10 | 500    | SRCNN  | tensorflow | 56.0   |
| cifar10 | 500    | SRCNN  | keras      | 25.9   |
| cifar10 | 500    | SRCNN  | pytorch    | 26.49  |

  
`tensorflow` 可能是因为数据集的问题导致 `psnr` 的计算会出现一些小的问题

因为数据集的使用问题，所以模型的训练是没有意义的。  
出于对`cifar`数据集的一个不了解，它是32*32的，但是我将它 bicubic 放大成了 128*128 作为 ground true。  
然后训练数据 从 32*32 resize 到 32*32 用邻近插值，然后又 bicubic 放大成 128*128 作为训练数据，这个是无效的训练。
所以训练效果直接爆炸。  
后续也不因数据集问题做更多的尝试和改进。整个内容当作对 `tensorflow > 2.0`  的一个入门尝试。

## References

👇是对`markdown`使用的一些了解

This repository is implementation of the ["Image Super-Resolution Using Deep Convolutional Networks"](https://arxiv.org/abs/1501.00092).

<center><img src=""></center>

## Train

The 91-image, Set5 dataset converted to HDF5 can be downloaded from the links below.

| Dataset | Scale | Type | Link |
|---------|-------|------|------|
| 91-image | 2 | Train | [Download](https://www.dropbox.com/s/2hsah93sxgegsry/91-image_x2.h5?dl=0) |
| 91-image | 3 | Train | [Download](https://www.dropbox.com/s/curldmdf11iqakd/91-image_x3.h5?dl=0) |
| 91-image | 4 | Train | [Download](https://www.dropbox.com/s/22afykv4amfxeio/91-image_x4.h5?dl=0) |
| Set5 | 2 | Eval | [Download](https://www.dropbox.com/s/r8qs6tp395hgh8g/Set5_x2.h5?dl=0) |
| Set5 | 3 | Eval | [Download](https://www.dropbox.com/s/58ywjac4te3kbqq/Set5_x3.h5?dl=0) |
| Set5 | 4 | Eval | [Download](https://www.dropbox.com/s/0rz86yn3nnrodlb/Set5_x4.h5?dl=0) |



* [liliumao/Tensorflow-srcnn](https://github.com/liliumao/Tensorflow-srcnn) 
  * - I referred to this repository which is same implementation using Matlab code and Caffe model.
<br>

* [carpedm20/DCGAN-tensorflow](https://github.com/carpedm20/DCGAN-tensorflow) 

