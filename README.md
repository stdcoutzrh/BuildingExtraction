## BuildingExtraction


## Introduction

* *IEEE GRSL:《SDSC-UNet: Dual Skip Connection ViT-based U-shaped Model for Building Extraction》.* 2023, Renhe Zhang, Qian Zhang and Guixu Zhang.

## Install

Open the folder **airs** using **Linux Terminal** and create python environment:
```
conda create -n airs python=3.8
conda activate airs

conda install pytorch==1.10.0 torchvision==0.11.0 torchaudio==0.10.0 cudatoolkit=11.3 -c pytorch -c conda-forge
pip install -r requirements.txt
```

## Training

```
python train_supervision.py -c ./config/inria/sdscunet.py
```

```
python train_supervision.py -c ./config/mass/sdscunet.py
```


## Testing

```
python building_seg_test.py -c ./config/inria/sdscunet.py -o /home/results/inria/sdscunet --rgb -t 'lr'
```

```
python building_seg_test.py -c ./config/mass/sdscunet.py -o /home/results/mass/sdscunet --rgb -t 'lr'
```



## Citation

If you find this project useful in your research, please consider citing our paper：

* [SDSC-UNet: Dual Skip Connection ViT-based U-shaped Model for Building Extraction]()

## Acknowledgement

- [BuildFormer](https://github.com/WangLibo1995/BuildFormer)
- [ShuntedTransformer](https://github.com/OliverRensu/Shunted-Transformer)
- [pytorch lightning](https://www.pytorchlightning.ai/)
- [timm](https://github.com/rwightman/pytorch-image-models)
- [pytorch-toolbelt](https://github.com/BloodAxe/pytorch-toolbelt)
- [ttach](https://github.com/qubvel/ttach)
- [catalyst](https://github.com/catalyst-team/catalyst)
- [mmsegmentation](https://github.com/open-mmlab/mmsegmentation)