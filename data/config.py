# config.py
import os.path

# gets home dir cross platform
home = os.path.expanduser("~")
# ddir = os.path.join(home,"data/VOCdevkit/")
ddir = '/localSSD/yyq/VOCdevkit0712'

# note: if you used our download scripts, this should be right
VOCroot = ddir # path to VOCdevkit root dir
COCOroot = "/localSSD/yyq/coco2015"
# default batch size
BATCHES = 32
# data reshuffled at every epoch
SHUFFLE = True
# number of subprocesses to use for data loading
WORKERS = 4


#SSD300 CONFIGS
# newer version: use additional conv11_2 layer as last layer before multibox layers
VOC_300 = {
    'feature_maps' : [38, 19, 10, 5, 3, 1],
    'refine': False,

    'min_dim' : 300,

    'in_channels_vgg': (512, 1024, 512, 256, 256, 256),

    'in_channels_res': (512, 1024, 512, 256, 256, 256),

    'num_anchors': (4, 6, 6, 6, 4, 4),

    'num_anchors_extra': (6, 6, 6, 6, 4, 4),

    'steps' : [8, 16, 32, 64, 100, 300],

    'steps_w' : [8, 16, 32, 64, 100, 300],

    'steps_h' : [8, 16, 32, 64, 100, 300],

    'min_sizes' : [30, 60, 111, 162, 213, 264],

    'max_sizes' : [60, 111, 162, 213, 264, 315],

    'aspect_ratios' : [[2], [2, 3], [2, 3], [2, 3], [2], [2]],

    'aspect_ratios_extra' : [[2, 3], [2, 3], [2, 3], [2, 3], [2], [2]],

    'variance' : [0.1, 0.2],

    'clip' : True,
    
    'use_extra_prior' : False,
}

channels_config = {
    "32_1" : [[32, 16, 16], [32, 16, 16], [32, 32, 16], [32, 16, 16], [32, 16, 16], [608, 1120, 608, 304]],
    "32_2" : [[32, 32, 32], [32, 32, 32], [32, 32, 32], [32, 32, 32], [32, 32, 32], [640, 1152, 640, 352]],
    "48" : [[48, 32, 32], [48, 32, 16], [48, 48, 32], [48, 32, 32], [48, 32, 32], [672, 1184, 672, 336]],
    "64" : [[64, 32, 32], [64, 32, 16], [64, 64, 32], [64, 32, 32], [64, 32, 32], [704, 1216, 704, 336]],
    "96" : [[96, 32, 32], [96, 32, 16], [96, 96, 32], [96, 32, 32], [96, 32, 32], [768, 1280, 768, 336]],
    "128" : [[128, 32, 32], [128, 32, 16], [128, 128, 32], [128, 32, 32], [128, 32, 32], [832, 1344, 832, 336]],
}

VOC_512= {
    'feature_maps' : [64, 32, 16, 8, 4, 2, 1],

    'min_dim' : 524,

    'refine': False,

    'in_channels_vgg': (512, 1024, 512, 256, 256, 256, 256),

    'in_channels_res': (512, 1024, 512, 256, 256, 256, 256),

    'num_anchors': (4, 6, 6, 6, 6, 4, 4),

    'num_anchors_extra': (6, 6, 6, 6, 6, 4, 4),

    'steps' : [8, 16, 32, 64, 128, 256, 512],

    'min_sizes'  : [35.84, 76.8, 153.6, 230.4, 307.2, 384.0, 460.8 ],

    'max_sizes'  : [76.8, 153.6, 230.4, 307.2, 384.0, 460.8, 537.6],

    'aspect_ratios' : [[2], [2, 3], [2, 3], [2, 3], [2,3], [2], [2]],

    'aspect_ratios_extra' : [[2,3], [2, 3], [2, 3], [2, 3], [2,3], [2], [2]],

    'variance' : [0.1, 0.2],

    'clip' : True,
    'use_extra_prior' : True,
}


COCO_300 = {
    'feature_maps' : [38, 19, 10, 5, 3, 1],

    'min_dim' : 300,

    'refine': False,

    'in_channels_vgg': (512, 1024, 512, 256, 256, 256),

    'in_channels_res': (512, 1024, 512, 256, 256, 256),

    'num_anchors': (4, 6, 6, 6, 4, 4),

    'num_anchors_extra': (6, 6, 6, 6, 4, 4),

    'steps' : [8, 16, 32, 64, 100, 300],

    'min_sizes' : [21, 45, 99, 153, 207, 261],

    'max_sizes' : [45, 99, 153, 207, 261, 315],

    'aspect_ratios' : [[2], [2, 3], [2, 3], [2, 3], [2], [2]],

    'aspect_ratios_extra' : [[2,3], [2, 3], [2, 3], [2, 3], [2], [2]],

    'variance' : [0.1, 0.2],

    'clip' : True,

    'use_extra_prior' : True,
}

COCO_512= {
    'feature_maps' : [64, 32, 16, 8, 4, 2, 1],

    'min_dim' : 512,

    'refine': False,

    'steps' : [8, 16, 32, 64, 128, 256, 512],

    'in_channels_vgg': (512, 1024, 512, 256, 256, 256, 256),

    'in_channels_res': (512, 1024, 512, 256, 256, 256, 256),

    'num_anchors': (4, 6, 6, 6, 6, 4, 4),

    'num_anchors_extra': (6, 6, 6, 6, 6, 4, 4),

    'min_sizes' : [20.48, 51.2, 133.12, 215.04, 296.96, 378.88, 460.8],

    'max_sizes' : [51.2, 133.12, 215.04, 296.96, 378.88, 460.8, 542.72],

    'aspect_ratios' : [[2], [2, 3], [2, 3], [2, 3], [2,3], [2], [2]],

    'aspect_ratios_extra' : [[2,3], [2, 3], [2, 3], [2, 3], [2,3], [2], [2]],

    'variance' : [0.1, 0.2],

    'clip' : True,

    'use_extra_prior' : True,
}
