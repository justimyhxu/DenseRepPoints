# Dense Reppoints: Representing Visual Objects with Dense Point Sets

We have released the "Dense RepPoints" on [Arixv](https://arxiv.org/abs/1912.11473).

Please checkout the [segmentation]() branch to reproduce the results in the "Dense Reppoints" on COCO object detection and instance segmentation. 


## Introduction

""Dense Reppoints"", which is based on ""RepPoints", is presented for ﬂexible and detailed modeling of object appearance and geometry.
In contrast to the coarse geometric localization and feature extraction of bounding boxes, 
'Dense RepPoints' adaptively distributes a dense set of attributed points to semantically and geometrically signiﬁcant positions on an object, 
providing informative cues for object analysis. 
Techniques are developed to address challenges related to supervised training for dense point sets and making this extensive representation computationally practical. 
The Dense RepPoints proves to perform significantly better than previous approaches or act as the ﬁrst effort in representing several irregular ﬁne-grained object shapes, 
substantially enlarging the research breadth for object shape analysis. 
In addition, this representation can represent multiple granularity of object structures, 
in which more ﬁne-grained geometric supervision benefit the coarse object detection task by 1.6 mAP on COCO detection benchmark.


<div align="center">
  <img src="demo/dense_reppoints.png" width="400px" />
  <p>Learning RepPoints in Object Detection.</p>
</div>

## Usage
a. Clone the repo and checkout segmentation branch:
```
git clone --recursive https://github.com/microsoft/RepPoints
git checkout segmentation
```
b. Download the COCO detection dataset, copy RepPoints src into mmdetection and install mmdetection. 
```
sh ./init.sh
```
c. Run experiments with a speicific configuration file:
```
./mmdetection/tools/dist_train.sh ${path-to-cfg-file} ${num_gpu} --validate
```
We give one example here:
```
./mmdetection/tools/dist_train.sh ./configs/dense_reppoints_r50_225pts_grid_sparse9_partial_fpn_1x.py 8 --validate
```

## Citing Dense RepPoints

```
@article{yang2019dense,
  title={Dense reppoints: Representing visual objects with dense point sets},
  author={Yang, Ze and Xu, Yinghao and Xue, Han and Zhang, Zheng and Urtasun, Raquel and Wang, Liwei and Lin, Stephen and Hu, Han},
  journal={arXiv preprint arXiv:1912.11473},
  year={2019}
}
```

## Results and models

The results on COCO 2017val are shown in the table below.

| Method | Backbone | Anchor | convert func | Lr schd | box AP | mask AP | Download |
| :----: | :------: | :-------: | :------: | :-----: | :----: | :------: | :------: |
| Dense RepPoints | R-50-FPN | none     | partial MinMax | 1x | 39.0| 29.2 | [model]() |


**Notes:**

- `R-xx`, `X-xx` denote the ResNet and ResNeXt architectures, respectively. 
- `DCN` denotes replacing 3x3 conv with the 3x3 deformable convolution in `c3-c5` stages of backbone.
- `none` in the `anchor` column means 2-d `center point` (x,y) is used to represent the initial object hypothesis. `single` denotes one 4-d anchor box (x,y,w,h) with IoU based label assign criterion is adopted. 
- `moment`, `partial MinMax`, `MinMax` in the `convert func` column are three functions to convert a point set to a pseudo box.
- `ms` denotes multi-scale training or multi-scale test.
- Note the results here are slightly different from those reported in the paper, due to framework change. While the original paper uses an [MXNet](https://mxnet.apache.org/) implementation, we re-implement the method in [PyTorch](https://pytorch.org/) based on mmdetection.

# Contributing
This project welcomes contributions and suggestions. Most contributions require you to agree to a Contributor License Agreement (CLA) declaring that you have the right to, and actually do, grant us the rights to use your contribution. For details, visit https://cla.microsoft.com.

When you submit a pull request, a CLA-bot will automatically determine whether you need to provide a CLA and decorate the PR appropriately (e.g., label, comment). Simply follow the instructions provided by the bot. You will only need to do this once across all repos using our CLA.

This project has adopted the Microsoft Open Source Code of Conduct. For more information see the Code of Conduct FAQ or contact opencode@microsoft.com with any additional questions or comments.
