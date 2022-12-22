from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import distutils.util
import os
import sys
from collections import defaultdict
from six.moves import xrange

# Use a non-interactive backend
import matplotlib
matplotlib.use('Agg')

import numpy as np
import cv2
import _init_paths  # pylint: disable=unused-import
import torch
import logging
import nn as mynn
from core.config import cfg, cfg_from_file, cfg_from_list, assert_and_infer_cfg
from core.test import im_detect_all
from modeling.model_builder import Generalized_RCNN
import datasets.dummy_datasets as datasets
from utils.timer import Timer
from utils.my_io import save_object
from prm.peak_response_mapping_3d import PeakResponseMapping_3d
from skimage import io

# OpenCL may be enabled by default in OpenCV3; disable it because it's not
# thread safe and causes unwanted GPU memory allocations.
cv2.ocl.setUseOpenCL(False)
logger = logging.getLogger(__name__)
from libtiff import TIFF

def parse_args():
    """Parse in command line arguments"""
    parser = argparse.ArgumentParser(description='Demonstrate mask-rcnn results')
    parser.add_argument(
        '--dataset', # required=True,
        help='training dataset')

    parser.add_argument(
        '--cfg', dest='cfg_file', # required=True,
        help='optional config file')
    parser.add_argument(
        '--set', dest='set_cfgs',
        help='set config keys, will overwrite config in the cfg_file',
        default=[], nargs='+')

    parser.add_argument(
        '--no_cuda', dest='cuda', help='whether use CUDA', action='store_false')

    parser.add_argument('--load_ckpt', help='path of checkpoint to load')
    parser.add_argument(
        '--load_detectron', help='path to the detectron weight pickle file')

    parser.add_argument(
        '--image_dir',
        help='directory to load images for demo')
    parser.add_argument(
        '--images', nargs='+',
        help='images to infer. Must not use with --image_dir')
    parser.add_argument(
        '--output_dir',
        help='directory to save demo results',
        default="infer_outputs")
    parser.add_argument(
        '--merge_pdfs', type=distutils.util.strtobool, default=False)

    args = parser.parse_args()

    return args

def main():
    """main function"""

    if not torch.cuda.is_available():
        sys.exit("Need a CUDA device to run the code.")

    args = parse_args()
    args.dataset = 'AD'
    args.cfg_file = '/home/zhiyi/Projects/AD22_5//configs/AD22/e2e_mask_rcnn_AD22.yaml'
    args.image_dir = '/data/zhiyi/AD_BrainImage/ExtraTest_221204/set2/imagepath.txt'
    args.images = []
    args.num_workers = 4
    args.batch_size = 4
    args.use_tfboard = True
    args.load_ckpt = '/home/zhiyi/Projects/AD22/model/AD_BrainImage/1026/model_step2999.pth'
    args.output_dir = '/data/zhiyi/AD_Output/ExtraTest_221204/model_step2999'

    print('Called with args:')
    print(args)

    assert args.image_dir or args.images
    assert bool(args.image_dir) ^ bool(args.images)

    if args.dataset.startswith("coco"):
        dataset = datasets.get_coco_dataset()
        cfg.MODEL.NUM_CLASSES = len(dataset.classes)
    elif args.dataset.startswith("keypoints_coco"):
        dataset = datasets.get_coco_dataset()
        cfg.MODEL.NUM_CLASSES = 2
    elif args.dataset == "AD":
        cfg.TEST.DATASETS = ('AD_det_seg_train',)
        cfg.MODEL.NUM_CLASSES = 2
    else:
        raise ValueError('Unexpected dataset name: {}'.format(args.dataset))

    print('load cfg from file: {}'.format(args.cfg_file))
    cfg_from_file(args.cfg_file)

    if args.set_cfgs is not None:
        cfg_from_list(args.set_cfgs)

    assert bool(args.load_ckpt) ^ bool(args.load_detectron), \
        'Exactly one of --load_ckpt and --load_detectron should be specified.'
    cfg.MODEL.LOAD_IMAGENET_PRETRAINED_WEIGHTS = False  # Don't need to load imagenet pretrained weights
    assert_and_infer_cfg()

    maskRCNN = Generalized_RCNN()

    if args.cuda:
        maskRCNN.cuda()

    maskRCNN.eval()
    if cfg.PRM_ON:
        model = PeakResponseMapping_3d(maskRCNN)
        model.inference()
    else:
        model = maskRCNN

    if args.load_ckpt:
        load_name = args.load_ckpt
        logger.info("loading checkpoint %s", load_name)
        checkpoint = torch.load(load_name, map_location=lambda storage, loc: storage)
        # net_utils.load_ckpt(model, checkpoint['model'])
        if cfg.MODEL.RPN_ONLY or (not cfg.MODEL.MASK_ON):
            model_dic = model.state_dict()
            pretrained_dict = {k: v for k, v in checkpoint['model'].items() if k in model_dic}
            model_dic.update(pretrained_dict)
            model.load_state_dict(model_dic)
        else:
            model.load_state_dict(checkpoint['model'])

        model = mynn.DataParallel(model, cpu_keywords=['im_info', 'roidb'],
                                 minibatch=True, device_ids=[0])  # only support single GPU
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    if args.image_dir:
        imglist = open(args.image_dir, 'r').readlines()
        imglist = [x.rstrip() for x in imglist]
    else:
        imglist = args.images
    num_images = len(imglist)


    for i in xrange(num_images):
        print('img', i)
        if not os.path.exists(imglist[i]):
            continue
        im = io.imread(imglist[i])
        assert im is not None
        if args.dataset == 'AD':
            im_name = imglist[i].split('/')[-1][:-4]
        if os.path.exists(os.path.join(args.output_dir,im_name)):
            continue
        timers = defaultdict(Timer)

        if cfg.PRM_ON:
            dets_total = np.empty((0, 7))

            # pre-process
            mask = im > 0
            mean_val = np.mean(im[mask])
            std_val = np.std(im[mask])
            im = (im - mean_val) / std_val
            #padding and cropping
            slices, height, width = im.shape
            orig_slices = im.shape[0]
            patch_size = cfg.TEST.IN_SIZE
            if slices < patch_size[0]:
                pad_s = np.int((patch_size[0] - slices) / 2)
                pad_e = patch_size[0] - slices - pad_s
                im = np.append(np.tile(im[0, :, :], (pad_s, 1, 1)), im, axis=0)
                im = np.append(im, np.tile(im[-1, :, :], (pad_e, 1, 1)), axis=0)
                slices = patch_size[0]
            else:
                pad_s = 0
            if args.dataset == 'AD':
                sidx = [0, 32, 64]
                hidx = [0, 96]
                widx = [0, 96]

            len_s = len(sidx)
            len_h = len(hidx)
            len_w = len(widx)
            for iss, s in enumerate(sidx):
                for ih, h in enumerate(hidx):
                    for iw, w in enumerate(widx):
                        num = iss * len_w * len_h + ih * len_w + iw
                        save_path = os.path.join(args.output_dir,
                                                 '{}'.format(im_name), 'instances', '{}'.format(num))
                        if not os.path.exists(save_path):
                            os.makedirs(save_path)
                        img_crop = im[s: s+patch_size[0], h: h+patch_size[1], w: w+patch_size[2]].copy().astype(np.float32)
                        blobs = {}
                        blobs['data'] = [torch.from_numpy(img_crop[np.newaxis, np.newaxis, :])]
                        im_scale = 1.0
                        im_info = np.hstack((img_crop.shape, im_scale))[np.newaxis, :]
                        blobs['im_info'] = [torch.from_numpy(im_info)]
                        blobs['im_scale'] = [im_scale]
                        aggregation, class_response_maps, valid_peak_list, peak_response_maps, dets = model(**blobs)
                        if dets is None:
                            continue
                        detsdata = dets.data.cpu().numpy()
                        dets_total = np.append(dets_total, detsdata, axis=0)
                        if pad_s:
                            off_set = np.array([0, 0, -pad_s, 0, 0, -pad_s, 0], dtype=np.float32)
                            dets_total = dets_total + off_set
                        prm = peak_response_maps.data.cpu().numpy()
                        for ch in range(prm.shape[0]):  # only support to process 4D feature maps [ch, z, y, x]
                            fm_ch = prm[ch, :]
                            fm_ch -= np.min(fm_ch)
                            fm_ch /= np.max(fm_ch)
                            fm_ch *= 255.
                            fm_ch = fm_ch.astype(np.uint8)
                            if pad_s:
                                fm_ch = fm_ch[pad_s : pad_s+orig_slices, :]
                            if not save_path is None:
                                image3D = TIFF.open(os.path.join(save_path, '{}.tif'.format(ch)), mode='w')
                                for k in range(fm_ch.shape[0]):
                                    image3D.write_image(fm_ch[k, :], compression='lzw', write_rgb=False)
                                image3D.close()
                        if not save_path is None:
                            np.save(os.path.join(save_path, 'dets.npy'), detsdata)
        else:
            cls_boxes, _, _ = im_detect_all(model, im, timers=timers)

            im_name, _ = os.path.splitext(os.path.basename(imglist[i]))

            if args.dataset == 'AD':
                det_file = os.path.join(args.output_dir, im_name + '.pkl')
            save_object(
                    dict(
                    all_boxes=cls_boxes,
                    # all_segms=cls_segms,
                    # all_keyps=all_keyps,
                    # cfg=cfg_yaml
                   ), det_file
            )
            logger.info('Wrote detections to: {}'.format(os.path.abspath(det_file)))


if __name__ == '__main__':
    main()
