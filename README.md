# SEDD-PCC: A Single Encoder-Dual Decoder Framework For End-To-End Learned Point Cloud Compression
### [**[paper]**](https://arxiv.org/abs/2505.16709)
### [**[project page]**]()
This repository contains source code for SEDD-PCC.
Coming Soon


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
pip install torch==1.7.1 torchvision==0.8.2 torchaudio==0.7.2
pip install torch==1.9.1+cu111 torchvision==0.10.1+cu111 torchaudio==0.9.1 -f https://download.pytorch.org/whl/torch_stable.html
```

* ## install MinkowskiEngine：
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

![trainingdata](https://github.com/kai0416s/ANF-Sparse-PCAC/blob/main/trainingdata.png)

- Testing：
8iVFB dataset(longdress, loot, redandblack, soldier, basketball_player, and dancer.)

![testingdata](https://github.com/kai0416s/ANF-Sparse-PCAC/blob/main/testingdata.png)

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
│                 ├──/R7.pth
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
]
```
- output path and check point location：
```
Output = '/0523'
Ckpt = '/final_result'
```
* The check point we have provide：
```
ckptdir_list = [
  './ckpts' + Ckpt + '/R0.pth',
  './ckpts' + Ckpt + '/R1.pth',
  './ckpts' + Ckpt + '/R2.pth',
  './ckpts' + Ckpt + '/R3.pth',
  './ckpts' + Ckpt + '/R4.pth',
  './ckpts' + Ckpt + '/R5.pth',
  './ckpts' + Ckpt + '/R6.pth',
  './ckpts' + Ckpt + '/R7.pth',
]
```
> R7.pth is the high rate.

Then you can run the test and get the result in folder 0523
and we also provide the experiment result.

## Test
```
python test.py
```

# Result
compared our approach with three methods, including two learning-based methods：YOGA, DeepPCC, JPEG Pleno VM4.1, V-PCCv22, and a traditional point cloud compression standard G-PCC(TMC13v23).


# Authors
These files are provided by National Chung Cheng University [Applied Computing and Multimedia Lab](https://chiang.ccu.edu.tw/index.php).

Please contact us (s69924246@gmail.com and yimmonyneath@gmail.com) if you have any questions.

