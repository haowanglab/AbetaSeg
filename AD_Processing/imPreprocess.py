import  os
import numpy as np
import tifffile
import math
from skimage.filters import gaussian
from imUtils import Slice2Stack

def crop_BrainImage(Image_root, Save_root, x, y, height, weight):
    """
        We used the matlab code for image split.
    """
    image_n = 256
    image_m = 256
    if not os.path.exists(Save_root):
        os.mkdir(Save_root)
    image_list = os.listdir(Image_root)
    start_n = 0
    end_n = start_n + image_n
    start_m = 0
    end_m = start_m + image_m
    for k in range(len(image_list)):
        count = 0
        image_path = os.path.join(Image_root,image_list[k])
        save_path = os.path.join(Save_root,image_list[k])
        if os.path.exists(save_path):
            continue
        image = tifffile.imread(image_path)
        crop_image = image[:, x:x+height,y:y+weight]
        # print(crop_image.shape)
        n = crop_image.shape[1]
        m = crop_image.shape[2]
        print(n,m)
        for i in range(math.floor(n/image_n)):
            start_m = 0
            end_m = start_m + image_m
            for j in range(math.floor((m/image_m))):
                image = crop_image[:,start_n:end_n,start_m:end_m]
                print(image.shape,start_n,end_n,start_m,end_m)
                start_m = start_m + image_m
                end_m = start_m + image_m;
                image_name = image_list[k].split('.')[0]+'_sample_'+str(count).zfill(4)+'.tif'
                save_path = os.path.join(Save_root,image_name)
                tifffile.imwrite(save_path,image,compress=2)
                count += 1
            start_n = start_n + image_n;
            end_n = start_n + image_n;

def Gaussian_filter_3D(image_root,save_root):
    """
        Image denoise using gaussian filter.
        Or you can use Fiji to do the same process.
    """
    print('---Image denoise start---')
    if not os.path.exists(save_root):
        os.mkdir(save_root)
    image_list = os.listdir(image_root)
    for i in range(len(image_list)):
        print(image_list[i])
        image_path = os.path.join(image_root,image_list[i])
        save_path = os.path.join(save_root,image_list[i])
        image = tifffile.imread(image_path)
        filter_image = (gaussian(image,2)*65536).astype('uint16')
        tifffile.imwrite(save_path,filter_image,compress=2)
    print('---finished---')



def Illumination_correction(image_root,save_root, issue=True):
    """
        Illumination correction for microscopy image data.
        For Golabal illumination, set issue = False.
        For Inner-issue illumination set issue = Ture.
    """
    if not os.path.exists(save_root):
        os.mkdir(save_root)
    image_list = os.listdir(image_root)
    for i in range(len(image_list)):
        image_path = os.path.join(image_root,image_list[i])
        save_path = os.path.join(save_root,image_list[i])
        image = tifffile.imread(image_path)
        if issue:
            print('--Issue Illumination correction Mode--')
            Intensity_slice_2 = [] #Intensity Issue
            for i in range(image.shape[0]):
                print(i)
                slice = image[i,:,:]
                Intensity_slice.append(np.mean(slice))
                if len(np.where(slice[:,:]>144)[0]) == 0:
                    Intensity_slice_2.append(0)
                else:
                    issue = np.where(slice[:, :] > 144)
                    Intensity_slice_2.append(np.mean(slice[issue]))
                print(Intensity_slice[i],Intensity_slice_2[i])
            middle = math.floor(image.shape[0] / 2)
            middle_intensity = Intensity_slice_2[middle]
            print(middle_intensity)
            for i in range(image.shape[0]):
                slice = image[i, :, :]
                if Intensity_slice_2[i] == 0:
                    image[i, :, :] = image[i, :, :]
                else:
                    image[i, :, :] = (slice / (Intensity_slice_2[i] / middle_intensity))
        elif issue == False:
            print('--Total Illumination correction Mode--')
            Intensity_slice = []  # Intensity total
            for i in range(image.shape[0]):
                print(i)
                slice = image[i,:,:]
                Intensity_slice.append(np.mean(slice))
            print(Intensity_slice)
            middle = math.floor(image.shape[0] / 2)
            middle_intensity = Intensity_slice[middle]
            for i in range(image.shape[0]):
                slice = image[i, :, :]
                if Intensity_slice[i] == 0:
                    image[i, :, :] = image[i, :, :]
                else:
                    image[i, :, :] = (slice / (Intensity_slice[i] / middle_intensity))
        tifffile.imwrite(save_path,image.astype(('uint16')),compress=2)
        with open('illumination.txt','a+') as f:
            f.writelines(str(Intensity_slice)+'\n')


"""
    This script is used for image preprocessing.
    Fill the path root brainsliceRoot = r' ' of your own dataset root.
"""
if __name__ == '__main__':
    # 1 Stack images(after reconstruction) for every 75 slices(300um),
    brainsliceRoot = r'D:\UserData\zhiyi\Data\AD_Data\3M_ABeta\Public_test\BrainImage'
    brainstackRoot = brainsliceRoot + '_stack'
    Slice2Stack(brainsliceRoot,brainstackRoot,75)

    # 2 Denoising by Gaussian filter
    brainimageRoot = brainstackRoot
    denoisebrainimageRoot = brainimageRoot + '_bl'
    Gaussian_filter_3D(brainimageRoot, denoisebrainimageRoot)

    # 3 crop the whole brain image to image blocks
    brainimageRoot = denoisebrainimageRoot
    cutimageSaveroot = brainimageRoot + '_split'
    x, y, height, weight = 729, 1059, 2816, 4096  # For 3M data
    crop_BrainImage(brainimageRoot, cutimageSaveroot, x, y, height, weight)

    # 4 Illumination correction for all image blocks
    uncorrectImageroot = cutimageSaveroot
    correctImageroot = uncorrectImageroot + '_ic'
    Illumination_correction(cutimageSaveroot, correctImageroot, True)
