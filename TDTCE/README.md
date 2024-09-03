## TDTCE (<u>T</u>raffic <u>D</u>iffusion models with <u>T</u>ransformers against NIDS in **<u>C</u>**onstrained <u>E</u>nvironments)

## <sub>Intro</sub>

TDTCE is a novel method for generating adversarial traffic instances. It achieves the creation of adversarial instances that closely mimic the distribution of benign ones, thereby significantly enhancing the ability to evade Network Intrusion Detection Systems (NIDS). To the best of our knowledge, this is the first study to utilize a Transformer-based diffusion model for generating adversarial instances specifically targeted at malicious traffic.


## Setup

We provide an `environment.yml` file that you can be used to create a Conda environment. If you only want 
to run pre-trained models locally on CPU, you can remove the `cudatoolkit` and `pytorch-cuda` requirements from the file.

```bash
conda env create -f environment.yml
conda activate TDiT
```


## Sampling TDiT
#### **Pre-trained TDiT checkpoints**

You can sample from our pre-trained TDiT models with `Sample.py`. 

```bash
python sample.py --image-size 16 --seed 1
```

#### **Custom TDiT checkpoints** 

If you've trained a new TDiT model with `Train.py` you can add the `--ckpt` argument to use your own checkpoint instead. 

```bash
python Sample.py --model TDiT_AdaLN-G --image-size 16 --ckpt /path/to/model.pt
```


## Training TDiT

We provide a training script for TDiT in [`train.py`](train.py). 

```bash
torchrun --nnodes=1 --nproc_per_node=N Train.py --model TDiT_AdaLN-G --data-path /path/to/imagenet/train
```

## Training & Sampling DDPM

We use DDPM as a kind of generative algorithm, and as a comparison with TDTCE, we also give its elaborate training and sampling process.

### Sample

#### 1. Use pre-training weights

Run `Sample.py` directly, and click enter in the terminal to generate a picture. The generated picture is located in `results/predict_out/predict_1x1_results.png` or `results/predict_out/predict_5x5_results.png`

```bash
python Sample.py
```

####  2. Use weights trained with your own data

Before sampling, you should follow the **training steps** to get `Diffusion_Traffic.pth`.

Run `Sample.py` directly, and click enter in the terminal to generate a picture. The generated picture is located in `results/predict_out/predict_1x1_results.png` or `results/predict_out/predict_5x5_results.png`

```bash
python Sample.py
```

### Train

Before training, put the desired image file in the datasets folder. Then, you should run `Listdata.py` under the root directory to generate `train_traffic.txt`, and ensure that there is file path content inside. Finally, you should run the `Train.py` for training, and the pictures generated during training can be viewed in the `results/train_out` folder.

```bash
python Listdata.py
python Train.py
```

