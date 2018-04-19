# from .voc import VOCDetection, AnnotationTransform, detection_collate, VOC_CLASSES
from .voc0712 import VOCDetection, AnnotationTransform, detection_collate, VOC_CLASSES
# from .coco import COCODetection
from .config import *
from .coco import *
from .data_augment import *

# class BaseTransform(object):
#     """Defines the transformations that should be applied to test PIL image
#         for input into the network

#     dimension -> tensorize -> color adj

#     Arguments:
#         resize (int): input dimension to SSD
#         rgb_means ((int,int,int)): average RGB of the dataset
#             (104,117,123)
#         swap ((int,int,int)): final order of channels
#     Returns:
#         transform (transform) : callable transform to be applied to test/val
#         data
#     """
#     def __init__(self, resize, rgb_means, swap=(2, 0, 1)):
#         self.means = rgb_means
#         self.resize = resize
#         self.swap = swap

#     # assume input is cv2 img for now
#     def __call__(self, img):

#         interp_methods = [cv2.INTER_LINEAR, cv2.INTER_CUBIC, cv2.INTER_AREA, cv2.INTER_NEAREST, cv2.INTER_LANCZOS4]
#         interp_method = interp_methods[0]
#         img = cv2.resize(np.array(img), (self.resize,
#                                          self.resize),interpolation = interp_method).astype(np.float32)
#         img -= self.means
#         # img = img[:, :, (2, 1, 0)]
#         img = img.transpose(self.swap)
#         return torch.from_numpy(img)    

