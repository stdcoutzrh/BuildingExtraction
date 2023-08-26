## BuildingExtraction


## Introduction

This is the code for our papers:

[R. Zhang, Q. Zhang and G. Zhang, "SDSC-UNet: Dual Skip Connection ViT-Based U-Shaped Model for Building Extraction," in IEEE Geoscience and Remote Sensing Letters, vol. 20, pp. 1-5, 2023, Art no. 6005005, doi: 10.1109/LGRS.2023.3270303.](https://ieeexplore.ieee.org/document/10108049)


[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/sdsc-unet-dual-skip-connection-vit-based-u/semantic-segmentation-on-inria-aerial-image)](https://paperswithcode.com/sota/semantic-segmentation-on-inria-aerial-image?p=sdsc-unet-dual-skip-connection-vit-based-u)


[R. Zhang, Z. Wan, Q. Zhang and G. Zhang, "DSAT-Net: Dual Spatial Attention Transformer for Building Extraction From Aerial Images," in IEEE Geoscience and Remote Sensing Letters, vol. 20, pp. 1-5, 2023, Art no. 6008405, doi: 10.1109/LGRS.2023.3304377.](https://ieeexplore.ieee.org/document/10221771)

  	
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/dsat-net-dual-spatial-attention-transformer/semantic-segmentation-on-inria-aerial-image)](https://paperswithcode.com/sota/semantic-segmentation-on-inria-aerial-image?p=dsat-net-dual-spatial-attention-transformer)


## Install

Open the folder **airs** using **Linux Terminal** and create python environment:
```
conda create -n airs python=3.8
conda activate airs

conda install pytorch==1.10.0 torchvision==0.11.0 torchaudio==0.10.0 cudatoolkit=11.3 -c pytorch -c conda-forge
pip install -r requirements.txt
```

## Training

SDSCUNet:
```
python train_supervision.py -c ./config/inria/sdscunet.py
```

```
python train_supervision.py -c ./config/mass/sdscunet.py
```

DSATNet:
```
python train_supervision.py -c ./config/inria/dsatnet.py
```

```
python train_supervision.py -c ./config/mass/dsatnet.py
```


## Testing

SDSCUNet:
```
python building_seg_test.py -c ./config/inria/sdscunet.py -o /home/results/inria/sdscunet --rgb -t 'lr'
```

```
python building_seg_test.py -c ./config/mass/sdscunet.py -o /home/results/mass/sdscunet --rgb -t 'lr'
```

DSATNet:
```
python building_seg_test.py -c ./config/inria/sdscunet.py -o /home/results/inria/dsatnet --rgb -t 'lr'
```

```
python building_seg_test.py -c ./config/mass/sdscunet.py -o /home/results/mass/dsatnet --rgb -t 'lr'
```



## Citation

If you find this project useful in your research, please consider citing our papers：

* R. Zhang, Q. Zhang and G. Zhang, "SDSC-UNet: Dual Skip Connection ViT-Based U-Shaped Model for Building Extraction," in IEEE Geoscience and Remote Sensing Letters, vol. 20, pp. 1-5, 2023, Art no. 6005005, doi: 10.1109/LGRS.2023.3270303.

```shell
@ARTICLE{10108049,
  author={Zhang, Renhe and Zhang, Qian and Zhang, Guixu},
  journal={IEEE Geoscience and Remote Sensing Letters}, 
  title={SDSC-UNet: Dual Skip Connection ViT-Based U-Shaped Model for Building Extraction}, 
  year={2023},
  volume={20},
  number={},
  pages={1-5},
  doi={10.1109/LGRS.2023.3270303}}
```

and:

* R. Zhang, Z. Wan, Q. Zhang and G. Zhang, "DSAT-Net: Dual Spatial Attention Transformer for Building Extraction From Aerial Images," in IEEE Geoscience and Remote Sensing Letters, vol. 20, pp. 1-5, 2023, Art no. 6008405, doi: 10.1109/LGRS.2023.3304377.

```shell
@ARTICLE{10221771,
  author={Zhang, Renhe and Wan, Zhechun and Zhang, Qian and Zhang, Guixu},
  journal={IEEE Geoscience and Remote Sensing Letters}, 
  title={DSAT-Net: Dual Spatial Attention Transformer for Building Extraction From Aerial Images}, 
  year={2023},
  volume={20},
  number={},
  pages={1-5},
  doi={10.1109/LGRS.2023.3304377}}
```

## Acknowledgement

- [BuildFormer](https://github.com/WangLibo1995/BuildFormer)
- [ShuntedTransformer](https://github.com/OliverRensu/Shunted-Transformer)
- [ConvNext](https://github.com/facebookresearch/ConvNeXt)
- [pytorch lightning](https://www.pytorchlightning.ai/)
- [timm](https://github.com/rwightman/pytorch-image-models)
- [pytorch-toolbelt](https://github.com/BloodAxe/pytorch-toolbelt)
- [ttach](https://github.com/qubvel/ttach)
- [catalyst](https://github.com/catalyst-team/catalyst)
- [mmsegmentation](https://github.com/open-mmlab/mmsegmentation)
