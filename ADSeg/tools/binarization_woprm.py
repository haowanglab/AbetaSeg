# -*- coding: utf-8 -*-
import _init_paths  # pylint: disable=unused-import
import numpy as np
from skimage import io
import os
from libtiff import TIFF
from skimage.measure import label
import utils.boxes_3d as box_utils_3d
from glob import glob
from otsu import otsu_py, otsu_py_2d_fast
from core.config import cfg
import pickle
prm_path = '/data/zhiyi/AD_Output/ExtraTest_221204/model_step2999'
save_path = '/data/zhiyi/AD_Output/ExtraTest_221204/Binary_model_step2999'
if not os.path.exists(save_path):
    os.makedirs(save_path)
imglist = open('/data/zhiyi/AD_BrainImage/ExtraTest_221204/set2/test.txt', 'r').readlines()
data_path = '/data/zhiyi/AD_BrainImage/ExtraTest_221204/set2/image'
imglist = [x.rstrip() for x in imglist]
nms_thresh = 0.2
score_thres = 0.9
for im_name in imglist:
    # if os.path.exists(os.path.join(det_path, '{}.pkl',format(im_name))) is not True:
    #     continue
    # if os.path.exists(os.path.join(save_path, '{}.tif'.format(im_name))):
    #     continue
    # if not os.path.exists(os.path.join(data_path, im_name, im_name+'.tif')):
    #     continue
    print('img', im_name)
    img = io.imread(os.path.join(data_path, im_name, im_name+'.tif'))
    seg = np.zeros(img.shape, dtype=np.uint16)
    mask_id = 0
    dets = np.empty((0, 7), dtype=np.float32)
    pkl_path = os.path.join(det_path, '{}.pkl'.format(im_name))
    f = open(pkl_path, 'rb')
    data = pickle.load(f)
    value = data['all_boxes'][1]
    print(value)
    value_filter = []
    for va in value:
        if va[6]>score_thres:
            value_filter.append(va)
    if len(value_filter) == 0:
        continue
    dets = np.concatenate((dets, value_filter))
    keep = box_utils_3d.nms_3d(dets, nms_thresh)
    dets = dets[keep, :].copy()
    sort_idex = dets[:, -1].argsort()[::-1]
    dets = dets[sort_idex, :]

    scores = np.zeros((0, 2), dtype=np.float32)
    bboxes = []
    for d, det in enumerate(dets):
        # mask_id += 1
        mask_id = 1
        [x1, y1, z1, x2, y2, z2] = det[:6].astype(int)
        # if z2-z1 < 5:
        #     continue
        # crop box-part image and prm
        box_img = img[z1: z2 + 1, y1: y2 + 1, x1: x2 + 1]
        img_bi = seg[:, :, :].copy()
        # normalize gray image and prm image into similar intensity range
        if box_img.size == 0:#zhiyi 7.13
            continue
        gray_max = np.max(box_img)
        box_img_float = box_img.astype(float)
        box_img_float = np.clip(box_img_float/gray_max*300, 0, 300)+30
        box_img = box_img_float.astype(np.uint16)

        # binarization using 1d otsu algorithm
        box_bi_thres = otsu_py(box_img)
        box_bi = np.where(box_img>box_bi_thres,1,0)

        # binarization using 2d otsu algorithm with gaussian filter
        # from skimage.filters import gaussian
        # box_img_gaussian = (gaussian(box_img,2)*65536).astype('uint16')
        # box_bi, _, _ = otsu_py_2d_fast(box_img, box_img_gaussian)

        # keep the biggest connected component
        labels = label(box_bi)
        largestCC = labels == (np.argsort(np.bincount(labels.flat)[1:])[-1] + 1)
        cc = largestCC.astype(np.uint16) * mask_id
        box_img_crop_bi_zeros = img_bi[z1: z2 + 1, y1: y2 + 1, x1: x2 + 1] == 0
        img_bi[z1: z2 + 1, y1: y2 + 1, x1: x2 + 1][box_img_crop_bi_zeros] = cc[box_img_crop_bi_zeros]
        seg[:, :, :] = img_bi.copy()

        if mask_id in np.unique(seg):
            scores = np.concatenate((scores, np.array([mask_id, det[-1]])[np.newaxis, :]), axis=0)
        print(mask_id,[x1,y1,z1,x2,y2,z2],[x2-x1,y2-y1,z2-z1])
        an = (str(mask_id) + ' ' + str([x1,y1,z1,x2,y2,z2]) + ' ' + str([x2-x1,y2-y1,z2-z1]) + '\n')
        bboxes.append(an)
        #print(bboxes[0])
    # save detection and segmentation results
    with open(os.path.join(save_path, '{}.txt'.format(im_name)),'w+') as f:
        for bbox in bboxes:
            print(bbox)
            f.writelines(bbox)
    np.save(os.path.join(save_path, '{}.npy'.format(im_name)), scores)
    image3D = TIFF.open(os.path.join(save_path, '{}.tif'.format(im_name)), mode='w')
    for k in range(seg.shape[0]):
        image3D.write_image(seg[k, :], compression='lzw', write_rgb=True)
    image3D.close()

