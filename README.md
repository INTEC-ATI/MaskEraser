# MaskeEraser
# DEMO

Youtube Coming Soon

# Features
The MaskEraser project uses the NVIDIA Jetson Development Kit series 
and WebCam to automatically remove masks.

This uses deep learning to remove only the masked part from 
the video of persons with a mask. 
The removed parts are then predicted and drawn in the AI's imagination.

このプロジェクトでは、NVIDIA Jetson Development Kitシリーズと
WebCamを使用して、 マスクを自動的に削除します。

これは、動画中のマスクをした人から、ディープラーニングを用いて「マスクを削除」
し、削除した部分をAIが「想像で描き直す」ものになります。

![result](https://github.com/so-jun/MaskEraser/blob/main/git.gif)
Based video by Pavel Danilyuk from Pexels

# Story
For the spread of COVID-19 around the world, 
there were many consequences. The photos you casually take 
with your smartphone are no exception to this. The photos taken 
after the spread of COVID-19 show family and friends wearing masks.

Now, it is difficult to go out without a mask.
A mask is important to prevent infection and transmission of 
COVID-19.

But on the other hand, 
wearing a mask makes it impossible for AI to recognize your face. 
It's not just the AI.

Humans also find it difficult to recognize the expressions of 
people who cover their mouths.

We should not be divided by COVID-19.
That is why I wanted to use technology to break down 
the walls (= masks) created by COVID-19.

世界中にコロナが広まったことにとって、様々な影響があった。
皆さんが何気なくスマホで撮影した写真もその例外では無い。
コロナが広まってから、撮影された写真には、マスクをした家族や友達が
映っている。

今や、マスクなしで外出するのは難しい。
コロナに感染しない、感染させないためにも、マスクは重要である。

しかし、一方でマスクをすることで、AIは顔認識することができなくなる。
それは、AIだけでは無い。

人間も口元を覆い隠した人の表情を窺い知ることが難しくなってしまっている。

我々はコロナによって、分断されるべきでは無い。
だから、コロナによって作られた壁（＝マスク）を技術で壊したいと思った。

# Requirement
### Hardware
- NVIDIA Jetson device (veirfied on Jetson Nano 4GB Development Kit and Jetson AGX Xavier Development Kit)

- Logcool 270 (USB Camera)   
https://www.amazon.com/Logitech-C270-720pixels-Black-webcam/dp/B01BGBJ8Y0/ref=sr_1_3?dchild=1&keywords=C270&qid=1605453031&sr=8-3
  
### Softwear
- Pytorch 1.8.0  
- Torchvision 0.9.0  
 https://forums.developer.nvidia.com/t/pytorch-for-jetson-version-1-8-0-now-available/72048
- JetPack 4.5.1  
 https://developer.nvidia.com/embedded/learn/get-started-jetson-nano-devkit#write
# Installation
## Make environment

### Install Pytorch
https://forums.developer.nvidia.com/t/pytorch-for-jetson-version-1-8-0-now-available/72048  

```
sudo apt-get install python3-pip libopenblas-base libopenmpi-dev python3-tk libavcodec-dev libavformat-dev libswscale-dev
pip3 install -U pip

# ---------------------Pytorch---------------------
wget https://nvidia.box.com/shared/static/p57jwntv436lfrd78inwl7iml6p13fzh.whl -O torch-1.8.0-cp36-cp36m-linux_aarch64.whl
python3 -m pip install pillow Cython numpy torch-1.7.0-cp36-cp36m-linux_aarch64.whl
# ---------------------Torchvision---------------------
git clone --branch v0.9.0 https://github.com/pytorch/vision torchvision
cd torchvision
export BUILD_VERSION=0.9.0
python3 setup.py install --user
```

### Download Pytorch_Retinaface weight
https://github.com/biubug6/Pytorch_Retinaface  
Download the weights from the URL. Then place them in your weight folder.
```
MaskeEraser
    ┗ weight ┳ mobilenet0.25_Final.pth
             ┗ mobilenetV1X0.25_pretrain.tar
```

## Run Training
### When training with your own data（Not required）

You can use data from [CelebA][1] or other sources.
And place the data in the following directory.

[1]:http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html  

```
[your data path]
    ┣ train ┳ image001.jpg
    ┃       ┣ image002.jpg
    ┃       ┣ ・・・
    ┃       ┗ imageXXX.jpg
    ┃ 
    ┗ val   ┳ image_val001.jpg   
            ┣ image_val002.jpg
            ┣ ・・・
            ┗ image_valXXX.jpg   
```
and you run this code.
```
CUDA_VISIBLE_DEVICES=＜GPU ID＞ python main.py --save_dir [your result path] --root [your data path] --mode train 
```
- __Attention__  
It takes about 2 days for Quadro RTX 8000.  
The training part has NOT been tested with Jetson Nano.

### IF download trained model
https://github.com/INTEC-ATI/MaskEraser/releases/download/v1.0/100000.pth  
Download the model from the URL. Then place them in your weight folder.
```
MaskeEraser
    ┗ weight ━ 100000.pth
```

## Run program
```
cd [MaskeEraser path]
python3  main_jetson.py
```
「撮影」・・・The image will be taken and saved in the ./picture folder.  
「終了」・・・Exit MaskerEraserPhotographer.

# References
This implementation is implemented with reference to [Image Inpainting for Irregular Holes Using Partial Convolutions [Liu+, arXiv2018]][3] and 
https://github.com/naoto0804/pytorch-inpainting-with-partial-conv.

Also, https://github.com/biubug6/Pytorch_Retinaface has been used for face recognition.

この実装は  
[Image Inpainting for Irregular Holes Using Partial Convolutions [Liu+, arXiv2018]][3]
と、その非公式実装である
https://github.com/naoto0804/pytorch-inpainting-with-partial-conv
を参考に実装されています。

また、顔認識には、
https://github.com/biubug6/Pytorch_Retinaface
が用いられています。

<!--[2]:https://arxiv.org/abs/1804.07723-->
[3]:https://github.com/NVIDIA/partialconv

# Author
*Jun So* Intec Inc.  
<so_jun@intec.co.jp>

# License
MaskerEraser is under MIT License.

