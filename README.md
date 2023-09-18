# Transtouch
We present a novel neural surface reconstruction method, called NeuS (pronunciation: /nuːz/, same as "news"), for reconstructing objects and scenes with high fidelity from 2D image inputs.

![](./static/intro_1_compressed.gif)
![](./static/intro_2_compressed.gif)

 ## [Paper](https://github.com/ritsu-a/transtouch) | [Data](https://github.com/ritsu-a/transtouch) | [Video](https://youtu.be/WbkK1TZfy3M)

This is the official repo for the implementation of **Transtouch: Learning Transparent Object Depth Sensing Through Sparse Touches** .

## Usage

#### Data Convention
The data is organized as follows:

```
<sceneId_viewId>
|-- meta.pkl    # scene parameters
|-- 1024_irL/R_*.png # images taken from left/right camera with different exposure
|-- label*.png  # labels of objects
|-- 1024_irL/R_real_templcn2_ps11_t0.005.png # generated binary pattern

```

We use split files in ./split_files to select scenes and views for train and test.

### Setup

Clone this repository

```shell
git clone https://github.com/ritsu-a/transtouch
cd transtouch
pip install -r requirements.txt
```

### Running

Replace $MODEL_PATH with the path where weights are stored.

- **Finetuning**

```shell
python transtouch/tune.py --cfg configs/baseline.yml -s TEST.WEIGHT $MODEL_PATH
```

- **Testing**

```shell
python transtouch/test.py --cfg configs/baseline.yml -s TEST.WEIGHT $MODEL_PATH

```



## Citation

Cite as below if you find this repository is helpful to your project:

```
```

## Acknowledgement

Some code snippets are borrowed from [Activezero2](https://github.com/callmeray/activezero2). The readme file is modified based on [NeuS](https://github.com/Totoro97/NeuS) Thanks for these great projects.