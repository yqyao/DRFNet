"""Adapted from:
    @longcw faster_rcnn_pytorch: https://github.com/longcw/faster_rcnn_pytorch
    @rbgirshick py-faster-rcnn https://github.com/rbgirshick/py-faster-rcnn
    Licensed under The MIT License [see LICENSE for details]
"""

from __future__ import print_function
import torch
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torchvision.transforms as transforms
from torch.autograd import Variable
from data import VOCroot
import torch.utils.data as data
from data import VOCroot, COCOroot, VOC_300, VOC_512, COCO_300, COCO_512, AnnotationTransform, COCOAnnotationTransform, COCODetection, VOCDetection, detection_collate, BaseTransform, VOC_CLASSES
from layers.functions import Detect
import sys
import time
import argparse
import numpy as np
import pickle
import cv2
from utils.nms_wrapper import nms
from utils.timer import Timer
from data.voc_eval import voc_eval
if sys.version_info[0] == 2:
    import xml.etree.cElementTree as ET
else:
    import xml.etree.ElementTree as ET

def str2bool(v):
    return v.lower() in ("yes", "true", "t", "1")

parser = argparse.ArgumentParser(description='Single Shot MultiBox Detection')
parser.add_argument('--trained_model', default='weights/drf_refine_vgg_48_VOC_epoches_80_0419_bgr.pth',
                    type=str, help='Trained state_dict file path to open')

parser.add_argument('-v', '--version', default='ssd_vgg',
                    help='dense_ssd or origin_ssd version.')
parser.add_argument('-s', '--size', default='300',
                    help='300 or 512 input size.')
parser.add_argument('-d', '--dataset', default='VOC',
                    help='VOC ,VOC0712++ or COCO dataset')
parser.add_argument('-c', '--channel_size', default='48',
                    help='channel_size 32_1, 32_2, 48, 64, 96, 128')
parser.add_argument('--save_folder', default='eval/', type=str,
                    help='File path to save results')
parser.add_argument('--confidence_threshold', default=0.01, type=float,
                    help='Detection confidence threshold')
parser.add_argument('--top_k', default=5, type=int,
                    help='Further restrict the number of predictions to parse')
parser.add_argument('--cuda', default=True, type=str2bool,
                    help='Use cuda to train model')
parser.add_argument('--voc_root', default=VOCroot, help='Location of VOC root directory')
parser.add_argument('--retest', default=False, type=bool,
                    help='test cache results')

args = parser.parse_args()

if not os.path.exists(args.save_folder):
    os.mkdir(args.save_folder)

if args.cuda and torch.cuda.is_available():
    torch.set_default_tensor_type('torch.cuda.FloatTensor')
else:
    torch.set_default_tensor_type('torch.FloatTensor')

def test_net(save_folder, net, detector, cuda, testset, transform,
             max_per_image=300, thresh=0.05):
    """Test a Fast R-CNN network on an image database."""
    num_images = len(testset)
    # all detections are collected into:
    #    all_boxes[cls][image] = N x 5 array of detections in
    #    (x1, y1, x2, y2, score)
    num_images = len(testset)
    num_classes = (21, 81)[args.dataset == 'COCO']
    all_boxes = [[[] for _ in range(num_images)]
                 for _ in range(num_classes)]
    # all_boxes = [[[] for _ in range(num_images)]
    #              for _ in range(len(VOC_CLASSES))]

    if not os.path.exists(save_folder):
        os.mkdir(save_folder)
    # timers
    _t = {'im_detect': Timer(), 'misc': Timer()}
    det_file = os.path.join(save_folder, 'detections.pkl')

    if args.retest:
        f = open(det_file,'rb')
        all_boxes = pickle.load(f)
        print('Evaluating detections')
        testset.evaluate_detections(all_boxes, save_folder)
        return

    for i in range(num_images):
        img = testset.pull_image(i)
        x = Variable(transform(img).unsqueeze(0), volatile=True)
        # print(x)
        if cuda:
            x = x.cuda()

        _t['im_detect'].tic()
        out = net(x)      # forward pass
        boxes, scores = detector.forward(out)
        detect_time = _t['im_detect'].toc()
        boxes = boxes[0]
        scores = scores[0]
        boxes = boxes.cpu().numpy()
        scores = scores.cpu().numpy()
        # scale each detection back up to the image
        scale = torch.Tensor([img.shape[1], img.shape[0],
                             img.shape[1], img.shape[0]]).cpu().numpy()
        boxes *= scale

        _t['misc'].tic()

        for j in range(1, num_classes):
            inds = np.where(scores[:, j] > thresh)[0]
            if len(inds) == 0:
                all_boxes[j][i] = np.empty([0, 5], dtype=np.float32)
                continue
            c_bboxes = boxes[inds]
            c_scores = scores[inds, j]
            c_dets = np.hstack((c_bboxes, c_scores[:, np.newaxis])).astype(
                np.float32, copy=False)
            if args.dataset == 'VOC':
                cpu = True
            else:
                cpu = False
            # print(len(c_dets))
            keep = nms(c_dets, 0.45, force_cpu=cpu)
            keep = keep[:50]
            c_dets = c_dets[keep, :]
            all_boxes[j][i] = c_dets
        if max_per_image > 0:
            image_scores = np.hstack([all_boxes[j][i][:, -1] for j in range(1,num_classes)])
            if len(image_scores) > max_per_image:
                image_thresh = np.sort(image_scores)[-max_per_image]
                for j in range(1, num_classes):
                    keep = np.where(all_boxes[j][i][:, -1] >= image_thresh)[0]
                    all_boxes[j][i] = all_boxes[j][i][keep, :]
        nms_time = _t['misc'].toc()
 
        if i % 20 == 0:
            print('im_detect: {:d}/{:d} {:.3f}s {:.3f}s'
                .format(i + 1, num_images, detect_time, nms_time))
            _t['im_detect'].clear()
            _t['misc'].clear()

    with open(det_file, 'wb') as f:
        pickle.dump(all_boxes, f, pickle.HIGHEST_PROTOCOL)
    print('Evaluating detections')
    testset.evaluate_detections(all_boxes, save_folder)

if __name__ == '__main__':
    # load net
    dataset_mean = (104, 117, 123)
    set_type = 'test'
    img_dim = (300,512)[args.size=='512']
    num_classes = (21, 81)[args.dataset == 'COCO']

    if args.dataset == 'VOC':
        # train_sets = [('0712', 'trainval')]
        cfg = (VOC_300, VOC_512)[args.size == '512']
    else:
        # train_sets = [('2014', 'train'), ('2014', 'valminusminival')]
        cfg = (COCO_300, COCO_512)[args.size == '512']

    if args.version == "ssd_vgg":
        from models.ssd.vgg_net import build_ssd
        print("ssd vgg")
    elif args.version == "ssd_res":
        from models.ssd.res_net import build_ssd
        print("ssd resnet")
    elif args.version == "drf_ssd_vgg":
        from models.drfssd.vgg_drfnet import build_ssd
        print("drf ssd vgg")
    elif args.version == "drf_ssd_res":
        from models.drfssd.resnet_drfnet import build_ssd
        print("drf ssd resnet")
    elif args.version == "drf_refine_vgg":
        from models.refine_drfssd.vgg_refine_drfnet import build_ssd
        cfg['refine'] = True
        print("refine drf_ssd vgg")
    else:
        print('Unkown version!')

    channel_size = args.channel_size
    if args.version.split("_")[0] == "drf":
        net = build_ssd(cfg, "test", img_dim, num_classes, channel_size)
    else:
        net = build_ssd(cfg, "test", img_dim, num_classes)
    # print(net.state_dict())
    state_dict = torch.load(args.trained_model)
    # In order to add resnet ssd , I modify origin code to unify different version.
    # vgg_ssd
    # from collections import OrderedDict
    # new_state_dict = OrderedDict()
    # for k, v in state_dict.items():
    #     head = k[:1]
    #     if head == 'v' or head == 'e' or head == "L":
    #         name = "extractor." + k
    #     else:
    #         name = k
    #     new_state_dict[name] = v
    # vgg drfssd
    # from collections import OrderedDict
    # new_state_dict = OrderedDict()
    # for k, v in state_dict.items():
    #     head = k[7:8]
    #     if head == 'v' or head == 'e' or head == "L" or head == "d":
    #         name = "extractor." + k[7:] # remove `module.`
    #         print(name)
    #     else:
    #         name = k[7:]
    #     new_state_dict[name] = v
    from collections import OrderedDict
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        head = k[:7]
        if head == 'module.':
            name = k[7:] # remove `module.`
        else:
            name = k
        new_state_dict[name] = v
    net.load_state_dict(new_state_dict)

    net.eval()
    print('Finished loading model!')
    # load data
    if args.dataset == 'VOC':
        dataset = VOCDetection(args.voc_root, [('0712', "2007_test")], None, AnnotationTransform())
    elif args.dataset == 'COCO':
        dataset = COCODetection(
            COCOroot, [('2014', 'minival')], None, COCOAnnotationTransform())
            #COCOroot, [('2015', 'test-dev')], None)

    if args.cuda:
        net = net.cuda()
        cudnn.benchmark = True
    # evaluation
    top_k = 200
    save_folder = os.path.join(args.save_folder, args.dataset)
    if args.version == "drf_refine_vgg":
        detector = Detect(num_classes, 0, cfg, use_arm=True)
    else:
        detector = Detect(num_classes, 0, cfg)
    test_net(save_folder, net, detector, args.cuda, dataset,
             BaseTransform(net.size, dataset_mean, (2, 0, 1)), top_k,
             thresh=args.confidence_threshold)
