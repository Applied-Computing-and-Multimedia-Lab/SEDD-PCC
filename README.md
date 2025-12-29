<h1 align="center">
SEDD-PCC: A Single Encoder-Dual Decoder Framework For End-To-End Learned Point Cloud Compression
</h1>

<p align="center">
<strong><a href="https://github.com/kai0416s">Kai-Hsiang Hsieh</a></strong>,
<strong><a href="mailto:yimmonyneath@gmail.com">Monyneath Yim</a></strong>,
<strong><a href="https://chiang.ccu.edu.tw/index.php">Jui-Chiu Chiang</a></strong>
</p>

<p align="center">
National Chung Cheng University, Taiwan
</p>


<p align="center">
  <a href=""><img src="https://img.shields.io/badge/Arxiv-2505.16709-b31b1b.svg?logo=arXiv" alt="arXiv"></a>
  <a href="https://github.com/Applied-Computing-and-Multimedia-Lab/SEDD-PCC/blob/main/LICENSE"><img src="https://img.shields.io/badge/License-MIT-yellow" alt="License"></a>
  <a href="https://applied-computing-and-multimedia-lab.github.io/SEDD-PCC/"><img src="https://img.shields.io/badge/project-SEDD-blue?logo=github" alt="Home Page"></a>
</p>


# [Arxiv 2025] Official Implementation for SEDD-PCC

A unified framework that employs a single encoder and two decoders to jointly compress and reconstruct both the geometry and attributes of point clouds.

With a simple yet effective network design, SEDD-PCC achieves highly competitive compression performance while maintaining efficiency.

# Coming Soon


# Abstract
To encode point clouds containing both geometry and attributes, most learning-based compression schemes treat geometry and attribute coding separately, employing distinct encoders and decoders. This not only increases computational complexity but also fails to fully exploit shared features between geometry and attributes. To address this limitation, we propose SEDD-PCC, an end-to-end learning-based framework for lossy point cloud compression that jointly compresses geometry and attributes. SEDD-PCC employs a single encoder to extract shared geometric and attribute features into a unified latent space, followed by dual specialized decoders that sequentially reconstruct geometry and attributes. Additionally, we incorporate knowledge distillation to enhance feature representation learning from a teacher model, further improving coding efficiency. With its simple yet effective design, SEDD-PCC provides an efficient and practical solution for point cloud compression. Comparative evaluations against both rule-based and learning-based methods demonstrate its competitive performance, highlighting SEDD-PCC as a promising AI-driven compression approach.

![architecture](https://github.com/Applied-Computing-and-Multimedia-Lab/SEDD-PCC/blob/main/images/SEDD.png)

# Requirments environment
* Create env：
```
conda create -n SEDDPCC python=3.7
conda activate SEDDPCC
conda install SEDDPCC devel -c anaconda
```
- Install torch：
```

pip install torch==1.9.1+cu111 torchvision==0.10.1+cu111 torchaudio==0.9.1 -f https://download.pytorch.org/whl/torch_stable.html
```

* ## Install MinkowskiEngine：
[MinkowskiEngine](https://github.com/NVIDIA/MinkowskiEngine)
* ## Requirements
Step 1. Install requirements:
```
pip install -r requirements.txt
```
Step 2. Use the torchacc
```
cd torchacc/
python3 setup.py install
cd ../
```

# dataset
* Training：
[ScanNet dataset](https://github.com/ScanNet/ScanNet), which is a large open-source dataset of indoor scenes.
>cube division with size 64* 64 *64. We randomly selected 50,000 cubes and used them for training.

![trainingdata](https://github.com/Applied-Computing-and-Multimedia-Lab/SEDD-PCC/blob/main/images/traindata.png)

- Testing：
8iVFB dataset(longdress, loot, redandblack, soldier, basketball_player, and dancer.)

![testingdata](https://github.com/Applied-Computing-and-Multimedia-Lab/SEDD-PCC/blob/main/images/testdata.png)

# check point download(coming soon)：
| check point  | [Link]()|
| ---------- | -----------|


## ❗ After data preparation, the overall directory structure should be：
```
│SEDD-PCC/
├──results/
├──output/
├──ckpts/
│   ├──/final_result/
│                 ├──/R5.pth
├──.......
```

# Training
* The default setting：

| High rate check point  | setting|
| ---------- | -----------|
| learning rate   | 8×10^(-4) which gradually decreases to 1×10^(-6)   |
| lamda   | 0.05   |
| Epoch   | 500  |

## Train
```
python train.py
```
- You need to change the check point location and then can train the low rate check point.
```
parser.add_argument("--init_ckpt", default='/SEDDPCC/ckpts/final_result/R7.pth')
```
# Testing

* input the orignal point cloud path：
```
filedir_list = [
  './testdata/8iVFB/longdress_vox10_1300.ply',
  './testdata/8iVFB/loot_vox10_1200.ply',
  './testdata/8iVFB/redandblack_vox10_1550.ply',
  './testdata/8iVFB/soldier_vox10_0690.ply',
  './testdata/Owlii/basketball_player_vox11_00000200.ply',
  './testdata/Owlii/dancer_vox11_00000001.ply'
]
```
- output path and check point location：
```
Output = '/1229'
Ckpt = '/final_result'
```
* The check point we have provide：
```
ckptdir_list = [
  './ckpts' + Ckpt + '/R1.pth',
  './ckpts' + Ckpt + '/R2.pth',
  './ckpts' + Ckpt + '/R3.pth',
  './ckpts' + Ckpt + '/R4.pth',
  './ckpts' + Ckpt + '/R5.pth',
]
```
> R5.pth is the high rate.

Then you can run the test and get the result in folder 1229
and we also provide the experiment result.

## Test
```
python test.py
```

# Result
compared our approach with three methods, including two learning-based methods：YOGA, DeepPCC, JPEG Pleno VM4.1, V-PCCv22, and a traditional point cloud compression standard G-PCC(TMC13v23).

# Acknowledgement
Thanks for awesome works [PCGCV2](https://github.com/NJUVISION/PCGCv2)

# Authors
These files are provided by National Chung Cheng University [Applied Computing and Multimedia Lab](https://chiang.ccu.edu.tw/index.php).

Please contact us (s69924246@gmail.com and yimmonyneath@gmail.com) if you have any questions.

