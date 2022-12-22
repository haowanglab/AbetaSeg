import re
import  os
import numpy as np
from math import ceil
import tifffile
import h5py
import shutil
import pickle


def H52tif(H5_root,Tiff_Save_Root):
    """
        Generate tiff file from H5 file.
    """
    if not os.path.exists(Tiff_Save_Root):
        os.mkdir(Tiff_Save_Root)
    for gt_name in os.listdir(H5_root):
        Path = os.path.join(H5_root, gt_name)
        SavePath = os.path.join(Tiff_Save_Root, gt_name.split('.')[0] + '.tif')
        if os.path.exists(SavePath):
            continue
        print(Path, '--->', SavePath)
        f = h5py.File(Path)
        for key in f.keys():
            print(f[key].name)
            print(f[key].shape)
            im_shape = f[key].shape
            image = np.zeros((im_shape[0], im_shape[1], im_shape[2]))
            count = 0
            for i in range(f[key].shape[0]):
                count = count + 1
                # print(f[key][i].shape)
                image[i, :, :] = np.where(f[key][i][:, :, 0] == 2, 0, f[key][i][:, :, 0])
            print(count)
            tifffile.imwrite(SavePath, image.astype(('uint16')), compress=2)


def Loadpkl(pkl_root,save_root,score_thres):
    """
        Generate bounding boxes from network intermediate output which format in '.pkl'.
        Predict score threshold can be set by user.
        Output: bounding boxes .txt file.
    """
    if not os.path.exists(save_root):
        os.mkdir(save_root)
    pkl_list = os.listdir(pkl_root)
    filter_bbox = []
    for i in range(len(pkl_list)):
        print(pkl_list[i])
        pkl_path = os.path.join(pkl_root, pkl_list[i])
        save_path = os.path.join(save_root, pkl_list[i].split('.')[0]+'.txt')
        f = open(pkl_path, 'rb')
        data = pickle.load(f)
        value = data['all_boxes'][1]
        list_sort = value[np.argsort(value[:, 6])][::-1]
        for bbox in list_sort:
            if bbox[6]>score_thres:
                #print(np.array2string(bbox[:6].astype('int'))[1:-1])
                pos = np.array2string(bbox[:6].astype('int'))[1:-1] + '\n'
                filter_bbox.append(pos)
        print(filter_bbox)
        with open(save_path,'w+') as f:
            f.writelines(filter_bbox)

def Loadnpy(npy_root, npy_save_root):
    """
        Load '.npy' which generated in peak response mapping process.
    """
    if not os.path.exists(npy_save_root):
        os.mkdir(npy_save_root)
    file_list = os.listdir(npy_root)
    print(file_list)
    pattern = r'(.*).npy'
    npy_list = []
    for i in range(len(file_list)):
        if re.match(pattern, file_list[i]):
            npy_list.append(file_list[i])
    for i in range(len(npy_list)):
        npy = np.load(os.path.join(npy_root,npy_list[i]))
        npy_txt = np.array2string(npy)
        save_path = os.path.join(npy_save_root, npy_list[i].split('.')[0]+'.txt')
        with open(save_path,'w+') as f :
            f.writelines(npy_txt)


def Output2PseudoLabel(Output_root , Pseudolabel_save_root, threshold = 0.5):
    """
        Network predictions can be re-used as the pseudo label.
        But the label confidence is relative low and NEED to be modified by user.
    """
    if not os.path.exists(Pseudolabel_save_root):
        os.mkdir(Pseudolabel_save_root)
    file_list = os.listdir(Output_root)
    pattern_npy = r'(.*).npy'
    pattern_bbox = r'(.*).txt'
    npy_list = []
    txt_list = []
    for i in range(len(file_list)):
        if re.match(pattern_npy, file_list[i]):
            npy_list.append(file_list[i])
        if re.match(pattern_bbox,file_list[i]):
            txt_list.append(file_list[i])
    for i in range(len(npy_list)):
        print(npy_list[i],txt_list[i])
        npy = np.load(os.path.join(Output_root, npy_list[i]))
        txt = open(os.path.join(Output_root, txt_list[i]),'r').readlines()
        save_txt = ['position_x position_y position_z radius_xy radius_z\n']
        for j in range(npy.shape[0]):
            if npy[j][1]>threshold:
                print(npy[j][1],txt[j])
                z = txt[j].strip().split(' ')
                x1,y1,z1,x2,y2,z2 = int(z[1][1:-1]),int(z[2][:-1]),int(z[3][:-1]),int(z[4][:-1]),int(z[5][:-1]),int(z[6][:-1])
                print(x1,y1,z1,x2,y2,z2)
                D_xy = np.where((x2 - x1) > (y2 - y1), (x2 - x1), (y2 - y1))
                D_z = z2 - z1
                r_xy, r_z = ceil(D_xy / 2), ceil(D_z / 2)
                x, y, z = ceil((x1 + x2) / 2), ceil((y1 + y2) / 2), ceil((z1 + z2) / 2)
                # Valid label
                # print(x, y, z, r_xy, r_z)
                save_txt.append(str(x) + ' ' + str(y) + ' ' + str(z) + ' ' + str(r_xy) + ' ' + str(r_z) + '\n')
        with open(os.path.join(Pseudolabel_save_root, txt_list[i]), 'w+') as f:
            for i in range(len(save_txt)):
                f.writelines(save_txt[i])

def rename_fully(Img_root, save_root):
    """
        Rename all the files generated from preprocess into a normalized structure.
            ---00000.tif
            ---00001.tif
            ---...
    """
    if not os.path.exists(save_root):
        os.mkdir(save_root)
    img_list = os.listdir(Img_root)
    for i in range(len(img_list)):
        shutil.copyfile(os.path.join(Img_root,img_list[i]),os.path.join(save_root,str(i).zfill(5)+'.tif'))

def Slice2Stack(Slice_root, Stack_save_root , Slice_num=75):
    """
        For 3D view, we generate image stack of 300 um thick sections from 2D slices.
        Physical resolution is 4 um here.
    """
    if not os.path.exists(Stack_save_root):
        os.mkdir(Stack_save_root)
    Slice_list = os.listdir(Slice_root)
    match = r'(.*)_C1.tif'
    Slice_c1_list = []
    for i in range(len(Slice_list)):
        if re.match(match, Slice_list[i]):
            Slice_c1_list.append(Slice_list[i])
    Stack_num = int(len(Slice_c1_list)/75)
    print(Stack_num)
    for i in range(Stack_num):
        save_image = np.zeros((75,4500,6300)).astype('uint16')
        for index in range((i)*75,(i+1)*75):
            print(Slice_c1_list[index])
            slice_path = os.path.join(Slice_root,Slice_c1_list[index])
            print(index%75)
            save_image[index%75] = tifffile.imread(slice_path).astype('uint16')
        save_file_name = 'AD_145M_5_' + str(i+1).zfill(3) +'_488nm_10X.tif'
        save_file_path = os.path.join(Stack_save_root, save_file_name)
        print(save_file_path)
        tifffile.imwrite(save_file_path,save_image, compress=2)


def Stack2Slice(Stack_root, Slice_root):
    """
        Generate image slices from 3D stack.
    """
    if not os.path.exists(Slice_root):
        os.mkdir(Slice_root)
    Stack_list = os.listdir(Stack_root)
    index = 0
    for i in range(len(Stack_list)):
        Stack_path = os.path.join(Stack_root,Stack_list[i])
        image = tifffile.imread(Stack_path)
        z, x, y = image.shape
        for j in range(z):
            save_path = os.path.join(Slice_root, 'Z'+str(index).zfill(5)+'_C1.tif')
            print(save_path)
            index += 1
            if os.path.exists(save_path):
                continue
            slice = image[j,:,:]
            tifffile.imwrite(save_path,slice,compress=2)

def BrainImage1to255(Slice_root, Slice_save_root):
    """
        Change the binary mask which pixel value = 1 to 255.
        In some 3D viewer, pixel value will influence the 3D render effect.
    """
    if not os.path.exists(Slice_save_root):
        os.mkdir(Slice_save_root)
    Slice_list = os.listdir(Slice_root)
    for k in range(len(Slice_list)):
        Slice_path = os.path.join(Slice_root, Slice_list[k])
        image = tifffile.imread(Slice_path)
        image = np.where(image[:,:] == 1 , 255, 0)
        save_path = os.path.join(Slice_save_root, Slice_list[k])
        print(save_path)
        tifffile.imwrite(save_path, image.astype('uint16'))