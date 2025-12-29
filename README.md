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
  <a href="https://arxiv.org/abs/2505.16709"><img src="https://img.shields.io/badge/Arxiv-2505.16709-b31b1b.svg?logo=arXiv" alt="arXiv"></a>
  <a href="https://github.com/Applied-Computing-and-Multimedia-Lab/SEDD-PCC/blob/main/LICENSE"><img src="https://img.shields.io/badge/License-MIT-yellow" alt="License"></a>
  <a href="https://applied-computing-and-multimedia-lab.github.io/SEDD-PCC/"><img src="https://img.shields.io/badge/project-SEDD-blue?logo=github" alt="Home Page"></a>
</p>


# [Arxiv 2025] Official Implementation for SEDD-PCC

 ⭐ A unified framework that employs a single encoder and two decoders to jointly compress and reconstruct both the geometry and attributes of point clouds.

 ✨ With a simple yet effective network design, SEDD-PCC achieves highly competitive compression performance while maintaining efficiency.

 ## News
* We are excited to announce that our more powerful unified model,  **MEGA-PCC**, will be released soon! Compared to **SEDD-PCC**, it achieves significantly better performance. Stay tuned!
* 2025.12.29 Release test code.


## Todo
- [x] ~~Release the Paper~~
- [x] ~~Release inference code~~
- [x] ~~Release checkpoint~~
- [ ] Release training code

# Abstract
To encode point clouds containing both geometry and attributes, most learning-based compression schemes treat geometry and attribute coding separately, employing distinct encoders and decoders. This not only increases computational complexity but also fails to fully exploit shared features between geometry and attributes. To address this limitation, we propose SEDD-PCC, an end-to-end learning-based framework for lossy point cloud compression that jointly compresses geometry and attributes. SEDD-PCC employs a single encoder to extract shared geometric and attribute features into a unified latent space, followed by dual specialized decoders that sequentially reconstruct geometry and attributes. Additionally, we incorporate knowledge distillation to enhance feature representation learning from a teacher model, further improving coding efficiency. With its simple yet effective design, SEDD-PCC provides an efficient and practical solution for point cloud compression. Comparative evaluations against both rule-based and learning-based methods demonstrate its competitive performance, highlighting SEDD-PCC as a promising AI-driven compression approach.

![architecture](https://github.com/Applied-Computing-and-Multimedia-Lab/SEDD-PCC/blob/main/static/images/SEDD.png)

# Experiment result
To comprehensively evaluate the performance of our SEDD-PCC, we compared it against standard MPEG benchmarks, including G-PCC TMC13 v23 Octree-RAHT and V-PCC TMC2 v22, following the MPEG Common Test Condition (CTC). Additionally, we compare our approach with several learning-based techniques for joint point cloud compression, specifically YOGA, DeepPCC, Unicorn, and JPEG Pleno. For a more comprehensive comparison, Table 1 presents the BD-BR (%) of our method and various benchmarks, using G-PCC as the anchor. Notably, our approach achieves substantial bitrate reductions compared to G-PCC, with average savings of 75.0% in D1-PSNR, 32.6% in Y-PSNR, and 33.2% in 1-PCQM. Furthermore, SEDD-PCC demonstrates superior coding performance when compared with JPEG Pleno for the Soldier sequence. Our method also shows a higher BD-rate saving in terms of 1-PCQM when compared to YOGA and DeepPCC, for the 11-bit sequences.

![RD](https://github.com/Applied-Computing-and-Multimedia-Lab/SEDD-PCC/blob/main/static/images/SEDDresult.svg)

![BDBR](https://github.com/Applied-Computing-and-Multimedia-Lab/SEDD-PCC/blob/main/static/images/BDBR.png)

# Complexity Evaluation
Table 2 presents the complexity analysis in terms of model size and encoding/decoding time. Due to availability constraints, only a subset of learning-based joint coding schemes is included. Our SEDD-PCC has a model size of 32.6 MB, making it significantly lighter than other methods. Its encoding and decoding process is exceptionally fast. Additionally, it eliminates the need for bit allocation between geometry and attribute coding for each point cloud, as well as recoloring, thereby reducing overall processing time. .

![complexSEDD](https://github.com/Applied-Computing-and-Multimedia-Lab/SEDD-PCC/blob/main/static/images/complexSEDD.png)

# ⚙️Requirments environment
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

![trainingdata](https://github.com/Applied-Computing-and-Multimedia-Lab/SEDD-PCC/blob/main/static/images/traindata.png)

- Testing：
8iVFB dataset(longdress, loot, redandblack, soldier, basketball_player, and dancer.)

![testingdata](https://github.com/Applied-Computing-and-Multimedia-Lab/SEDD-PCC/blob/main/static/images/testdata.png)

# check point download (ready)：
| check point  | [Link](https://drive.google.com/drive/folders/1XYFupj8nfwjPdi1SZTJlGoloq-U7O5us?usp=sharing)|
| ---------- | -----------|


## ❗ After data preparation, the overall directory structure should be：
```
│SEDD-PCC/
├──PCAGC/
├──stage3Finetune/
│   ├──/output/
│   ├──/ckpts/TG/
│              ├──/R6.pth (High rate)
├──.......
```

# Training
* The default setting：

Epoch: 60

The learning rate is initialized at 8e-5 and halved every 20 epochs until it decreases to 2e-5.

| Parameter | R6 | R5 | R4 | R3 | R2 | R1 |
|----------------------|----|----|----|----|----|----|
| lamda_A               | 0.03 | 0.04 | 0.04 | 0.05 | 0.05 | 0.05 |
| lamda_G               | 6    | 4    | 4    | 8    | 12   | 20   |
| lamda_t               | 0.5  | 0.25 | 0.125| 0.05 | 0.015| 0.005|

## Train
```
python train.py
```
- You need to change the check point location and then can train the low rate check point.
```
parser.add_argument("--init_ckpt", default='/SEDDPCC/stage3Finetune/ckpts/TG/R6.pth')
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
Ckpt = '/TG/'
```
* The check point we have provide：
```
ckptdir_list = [
  './SEDDPCC/stage3Finetune/ckpts/TG/R1.pth',
  './SEDDPCC/stage3Finetune/ckpts/TG/R2.pth',
  './SEDDPCC/stage3Finetune/ckpts/TG/R3.pth',
  './SEDDPCC/stage3Finetune/ckpts/TG/R4.pth',
  './SEDDPCC/stage3Finetune/ckpts/TG/R5.pth',
  './SEDDPCC/stage3Finetune/ckpts/TG/R6.pth'
]
```
> R6.pth is the high rate.

Then you can run the test and get the result in folder 1229
and we also provide the experiment result.

- PCAGC consists of three stages: the first stage performs attribute-only compression, the second stage performs geometry compression, and the third stage performs joint compression using the checkpoint from stage3Finetune.

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

