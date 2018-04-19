import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1,0"
import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
import torch.nn.init as init
import argparse
from torch.autograd import Variable
import torch.utils.data as data
from data import VOCroot, COCOroot, VOC_300, VOC_512, COCO_300, COCO_512, AnnotationTransform, COCOAnnotationTransform, COCODetection, VOCDetection, detection_collate, BaseTransform, VOC_CLASSES,preproc
from utils.augmentations import SSDAugmentation
from layers.modules import MultiBoxLoss
from layers.functions import Detect
import numpy as np
import time
import os 
import sys


def str2bool(v):
    return v.lower() in ("yes", "true", "t", "1")

parser = argparse.ArgumentParser(
    description='Receptive Field Block Net Training')
parser.add_argument('-v', '--version', default='ssd_vgg',
                    help='')
parser.add_argument('-s', '--size', default='300',
                    help='300 or 512 input size.')
parser.add_argument('-d', '--dataset', default='VOC',
                    help='VOC or COCO dataset')
parser.add_argument('-c', '--channel_size', default='48',
                    help='channel_size')
parser.add_argument(
    '--basenet', default='./weights/vgg16_reducedfc.pth', help='pretrained base model')
parser.add_argument('--jaccard_threshold', default=0.5,
                    type=float, help='Min Jaccard index for matching')
parser.add_argument('-b', '--batch_size', default=32,
                    type=int, help='Batch size for training')
parser.add_argument('--num_workers', default=4,
                    type=int, help='Number of workers used in dataloading')
parser.add_argument('--cuda', default=True,
                    type=bool, help='Use cuda to train model')
parser.add_argument('--lr', '--learning-rate',
                    default=4e-3, type=float, help='initial learning rate')
parser.add_argument('--ngpu', default=2, type=int, help='gpus')
# parser.add_argument('--finetune', default='voc', help='finetune from coco?')
# parser.add_argument('--finetune_weights', default='./weights/0211_Final_bgr_dense_ssd_48_COCO_300.pth', help='finetune from coco')
parser.add_argument('--momentum', default=0.9, type=float, help='momentum')
parser.add_argument(
    '--resume_net', default=None, help='resume net for retraining')
parser.add_argument('--resume_epoch', default=0,
                    type=int, help='resume iter for retraining')
parser.add_argument('-max','--max_epoch', default=250,
                    type=int, help='max epoch for retraining')
parser.add_argument('--weight_decay', default=5e-4,
                    type=float, help='Weight decay for SGD')
parser.add_argument('--gamma', default=0.1,
                    type=float, help='Gamma update for SGD')
parser.add_argument('--log_iters', default=True,
                    type=bool, help='Print the loss at each iteration')
parser.add_argument('--save_folder', default='./weights/',
                    help='Location to save checkpoint models')
args = parser.parse_args()

if not os.path.exists(args.save_folder):
    os.mkdir(args.save_folder)

if args.cuda and torch.cuda.is_available():
    torch.set_default_tensor_type('torch.cuda.FloatTensor')
else:
    torch.set_default_tensor_type('torch.FloatTensor')

datasets_dict = {"VOC": [('0712', '0712_trainval')],
            "VOC0712++": [('0712', '0712_trainval_test')],
            "VOC2012" : [('2012', '2012_trainval')],
            "COCO": [('2014', 'train'), ('2014', 'valminusminival')],
            "VOC2007": [('0712', "2007_test")],
            "COCOval": [('2014', 'minival')]}

p = 0.6
img_dim = (300,512)[args.size=='512']
bgr_means = (104, 117, 123)
num_classes = (21, 81)[args.dataset == 'COCO']
batch_size = args.batch_size
weight_decay = 0.0005
gamma = 0.1
momentum = 0.9

dataset_name = args.dataset
if dataset_name[0] == "V":
    cfg = (VOC_300, VOC_512)[args.size == '512']
    train_dataset = VOCDetection(VOCroot, datasets_dict[dataset_name], SSDAugmentation(img_dim, bgr_means), AnnotationTransform(), dataset_name)
    # train_dataset = VOCDetection(VOCroot, datasets_dict[dataset_name],    preproc(img_dim, bgr_means, p), AnnotationTransform())    
    test_dataset = VOCDetection(VOCroot, datasets_dict["VOC2007"], None, AnnotationTransform(), dataset_name)
elif dataset_name[0] == "C":
    train_dataset = COCODetection(COCOroot, datasets_dict[dataset_name], SSDAugmentation(img_dim, bgr_means), COCOAnnotationTransform(), dataset_name)    
    test_dataset = COCODetection(COCOroot, datasets_dict["COCOval"], None, COCOAnnotationTransform(), dataset_name)
    cfg = (COCO_300, COCO_512)[args.size == '512']
else:
    print('Unkown dataset!')    


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
else:
    print('Unkown version!')

detector = Detect(num_classes, 0, cfg)
channel_size = args.channel_size
if args.version.split("_")[0] == "drf":
    net = build_ssd(cfg, "train", img_dim, num_classes, channel_size)
else:
    net = build_ssd(cfg, "train", img_dim, num_classes)

print(net)

if args.resume_net == None:
    net.load_weights(args.basenet)
else:
# load resume network
    state_dict = torch.load(args.resume_net)
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
    print('Loading resume network...')

if args.ngpu > 1:
    # net = torch.nn.DataParallel(net, device_ids=list(range(args.ngpu)))
    net = torch.nn.DataParallel(net)

if args.cuda:
    net.cuda()
    cudnn.benchmark = True

optimizer = optim.SGD(net.parameters(), lr=args.lr,
                      momentum=args.momentum, weight_decay=args.weight_decay)

criterion = MultiBoxLoss(num_classes, 0.5, True, 0, True, 3, 0.5, False, args.cuda)

def train():
    net.train()
    # loss counters
    loc_loss = 0  # epoch
    conf_loss = 0
    epoch = 0 + args.resume_epoch
    print('Loading Dataset...')

    epoch_size = len(train_dataset) // args.batch_size
    max_iter = args.max_epoch * epoch_size

    # stepvalues_VOC = (150 * epoch_size, 200 * epoch_size, 250 * epoch_size)
    stepvalues_VOC = (150 * epoch_size, 200 * epoch_size, 250 * epoch_size)
    stepvalues_COCO = (90 * epoch_size, 120 * epoch_size, 140 * epoch_size)
    stepvalues = (stepvalues_VOC, stepvalues_COCO)[args.dataset=='COCO']
    print('Training',args.version, 'on', train_dataset.name)
    step_index = 0

    if args.resume_epoch > 0:
        start_iter = args.resume_epoch * epoch_size
    else:
        start_iter = 0

    lr = args.lr
    for iteration in range(start_iter, max_iter):
        if iteration % epoch_size == 0:
            # create batch iterator
            batch_iterator = iter(data.DataLoader(train_dataset, batch_size,
                                                  shuffle=True, num_workers=args.num_workers, collate_fn=detection_collate))
            loc_loss = 0
            conf_loss = 0
            if (epoch % 10 == 0 and epoch > 0) or (epoch % 5 == 0 and epoch > 200):
                torch.save(net.state_dict(), args.save_folder+args.version+'_'+channel_size+'_'+args.dataset + '_epoches_'+
                           repr(epoch) + "_0417_bgr"+ '.pth')
            if (epoch % 50 == 0 and epoch > 0) or (epoch % 5 == 0 and epoch > 230):
                net.eval()
                save_folder = "./eval/"
                print("eval epoch: ", epoch)
                top_k = (300, 200)[args.dataset == 'COCO']
                test_net(save_folder, net, detector, args.cuda, test_dataset,BaseTransform(net.size, bgr_means, (2, 0, 1)),top_k,thresh=0.01)              

            epoch += 1

        load_t0 = time.time()
        if iteration in stepvalues:
            step_index += 1
        lr = adjust_learning_rate(optimizer, args.gamma, epoch, step_index, iteration, epoch_size)


        # load train data
        images, targets = next(batch_iterator)
        if args.cuda:
            images = Variable(images.cuda())
            targets = [Variable(anno.cuda(),volatile=True) for anno in targets]
        else:
            images = Variable(images)
            targets = [Variable(anno, volatile=True) for anno in targets]
        # forward
        t0 = time.time()
        out = net(images)
        # backprop
        t1 = time.time()
        optimizer.zero_grad()
        loss_l, loss_c = criterion(out, targets)

        loss = loss_l + loss_c
        loss.backward()

        optimizer.step()

        loc_loss += loss_l.data[0]
        conf_loss += loss_c.data[0]
        load_t1 = time.time()

        if iteration % 10 == 0:
            print('Epoch:' + repr(epoch) + ' || epochiter: ' + repr(iteration % epoch_size) + '/' + repr(epoch_size)
                  + '|| Totel iter ' +
                  repr(iteration) + ' || L: %.4f C: %.4f||' % (
                loss_l.data[0],loss_c.data[0]) + 
                'iteration time: %.4f sec. ||' % (load_t1 - load_t0) + 'LR: %.8f' % (lr))

    torch.save(net.state_dict(), args.save_folder +
               '0417_Final_bgr_' + args.version +'_'+channel_size+'_' + args.dataset+"_"+args.size+ '.pth')


def test_net(save_folder, net, detector, cuda, testset, transform,
             max_per_image=300, thresh=0.05):
    """Test a Fast R-CNN network on an image database."""
    num_images = len(testset)
    # all detections are collected into:
    #    all_boxes[cls][image] = N x 5 array of detections in
    #    (x1, y1, x2, y2, score)
    num_images = len(testset)
    num_classes = (21, 81)[testset.name == 'COCOval']
    all_boxes = [[[] for _ in range(num_images)]
                 for _ in range(num_classes)]

    if not os.path.exists(save_folder):
        os.mkdir(save_folder)
    # timers
    _t = {'im_detect': Timer(), 'misc': Timer()}
    det_file = os.path.join(save_folder, 'detections.pkl')

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
            if testset.name == 'VOC2007':
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

    with open(det_file, 'wb') as f:
        pickle.dump(all_boxes, f, pickle.HIGHEST_PROTOCOL)
    print('Evaluating detections')
    testset.evaluate_detections(all_boxes, save_folder)

def adjust_learning_rate(optimizer, gamma, epoch, step_index, iteration, epoch_size):
    """Sets the learning rate 
    # Adapted from PyTorch Imagenet example:
    # https://github.com/pytorch/examples/blob/master/imagenet/main.py
    """
    if epoch < 3:
        # lr = 1e-6 + (args.lr-1e-6) * iteration / (epoch_size * 5) 
        lr = 0.001
    else:
        lr = args.lr * (gamma ** (step_index))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr


if __name__ == '__main__':
    train()
