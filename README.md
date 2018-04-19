# SSD: Single Shot MultiBox Object Detector, in PyTorch
A [PyTorch](http://pytorch.org/) implementation of Dense Receptive Field for Object Detection (accepted by ICPR2018)

<img align="right" src= "https://github.com/amdegroot/ssd.pytorch/blob/master/doc/ssd.png" height = 400/>

### Table of Contents
- <a href='#installation'>Installation</a>
- <a href='#datasets'>Datasets</a>
- <a href='#training'>Train</a>
- <a href='#evaluation'>Evaluate</a>
- <a href='#performance'>Performance</a>
- <a href='#references'>Reference</a>

&nbsp;
&nbsp;
&nbsp;
&nbsp;

## Installation
- Install [PyTorch-0.3.1](http://pytorch.org/)  by selecting your environment on the website and running the appropriate command.
- Clone this repository.
  * Note: We currently only support Python 3+.
- Then download the dataset by following the [instructions](#download-voc2007-trainval--test) below.
- Compile the nms and coco tools:
```shell
cd DRFNet
./make.sh
```

Note*: Check you GPU architecture support in utils/build.py, line 131. Default is:

```Shell
'nvcc': ['-arch=sm_52',

```

## Datasets
To make things easy, we provide a simple VOC dataset loader that inherits `torch.utils.data.Dataset` making it fully compatible with the `torchvision.datasets` [API](http://pytorch.org/docs/torchvision/datasets.html).

### VOC Dataset
##### Download VOC2007 trainval & test

```Shell
# specify a directory for dataset to be downloaded into, else default is ~/data/
sh data/scripts/VOC2007.sh # <directory>
```

##### Download VOC2012 trainval

```Shell
# specify a directory for dataset to be downloaded into, else default is ~/data/
sh data/scripts/VOC2012.sh # <directory>
```

##### Merge VOC2007 and VOC2012

```Shell
move all images in VOC2007 and VOC2012 into VOCROOT/VOC0712/JPEGImages
move all annotations in VOC2007 and VOC2012 into VOCROOT/VOC0712/JPEGImages/Annotations
rename and merge some txt VOC2007 and VOC2012 ImageSets/Main/*.txt to VOCROOT/VOC0712/JPEGImages/ImageSets/Main/*.txt
the merged txt list as follows:
2012_test.txt, 2007_test.txt, 0712_trainval_test.txt, 2012_trainval.txt, 0712_trainval.txt

```
### COCO Dataset
Install the MS COCO dataset at /path/to/coco from [official website](http://mscoco.org/), default is ~/data/COCO. Following the [instructions](https://github.com/rbgirshick/py-faster-rcnn/blob/77b773655505599b94fd8f3f9928dbf1a9a776c7/data/README.md) to prepare *minival2014* and *valminusminival2014* annotations. All label files (.json) should be under the COCO/annotations/ folder. It should have this basic structure
```Shell
$COCO/
$COCO/cache/
$COCO/annotations/
$COCO/images/
$COCO/images/test2015/
$COCO/images/train2014/
$COCO/images/val2014/
```
*UPDATE*: The current COCO dataset has released new *train2017* and *val2017* sets which are just new splits of the same image sets. 


## Training
- First download the fc-reduced [VGG-16](https://arxiv.org/abs/1409.1556) PyTorch base network weights at:              https://s3.amazonaws.com/amdegroot-models/vgg16_reducedfc.pth
- ResNet pre-trained basenet weight file is available at [ResNet50](https://download.pytorch.org/models/resnet50-19c8e357.pth), [ResNet101](https://download.pytorch.org/models/resnet101-5d3b4d8f.pth), [ResNet152](https://download.pytorch.org/models/resnet152-b121ed2d.pth)
- By default, we assume you have downloaded the file in the `DRFNet/weights` dir:

```Shell
mkdir weights
cd weights
wget https://s3.amazonaws.com/amdegroot-models/vgg16_reducedfc.pth
wget https://download.pytorch.org/models/resnet50-19c8e357.pth
wget https://download.pytorch.org/models/resnet101-5d3b4d8f.pth
wget https://download.pytorch.org/models/resnet152-b121ed2d.pth
```

- To train DRFNet using the train script simply specify the parameters listed in `train.py` as a flag or manually change them.

```Shell
python train.py -v drf_ssd_vgg
```

- Note:
    * -d: choose datasets, VOC or COCO, VOC2012(voc12 trainval),VOC0712++(0712 trainval + 07test)
    * -v choose backbone version, ssd_vgg, ssd_res, drf_ssd_vgg, drf_ssd_res
    * s: image size, 300 or 512
    * You can pick-up training from a checkpoint by specifying the path as one of the training parameters (again, see train.py for options)
  
- To evaluate a trained network:

```Shell
python eval.py -v drf_ssd_vgg
```

You can specify the parameters listed in the `eval.py` file by flagging them or manually changing them.  


<img align="left" src= "https://github.com/yqyao/DRFNet/blob/master/data/drf_net.jpg">

## Performance

#### VOC2007 Test

##### mAP
we retrained some models, so it's different from the origin paper
size = 300
| ssd | drf_32 | drf_48 | drf_64 | drf_96 | drf_128 |
|:-:|:-:|:-:|:-:|:-:|:-:|
| 77.2 % | 79.87 % | 79.93% | 79.73 % | 79.38% | 79.65 % |

##### Evaluation report for the best current version

VOC07 metric? Yes

AP for aeroplane = 0.8579<br />
AP for bicycle = 0.8615<br />
AP for bird = 0.7786<br />
AP for boat = 0.7202<br />
AP for bottle = 0.5850<br />
AP for bus = 0.8788<br />
AP for car = 0.8712<br />
AP for cat = 0.8849<br />
AP for chair = 0.6612<br />
AP for cow = 0.8702<br />
AP for diningtable = 0.7796<br />
AP for dog = 0.8577<br />
AP for horse = 0.8750<br />
AP for motorbike = 0.8778<br />
AP for person = 0.8046<br />
AP for pottedplant = 0.5582<br />
AP for sheep = 0.7952<br />
AP for sofa = 0.8041<br />
AP for train = 0.8800<br />
AP for tvmonitor = 0.7847<br />
Mean AP = 0.7993<br />


##### FPS
**GTX 1080 Ti:** ~70 FPS 


## References
- Wei Liu, et al. "SSD: Single Shot MultiBox Detector." [ECCV2016]((http://arxiv.org/abs/1512.02325)).
- [Original Implementation (CAFFE)](https://github.com/weiliu89/caffe/tree/ssd)
- A list of other great SSD ports that were sources of inspiration (especially the Chainer repo): 
  * [ssd.pytorch]((https://github.com/amdegroot/ssd.pytorch)),
    [RFBNet](https://github.com/ruinmessi/RFBNet)
    [Chainer](https://github.com/Hakuyume/chainer-ssd),
    [torchcv](https://github.com/kuangliu/torchcv)
  ) 

