from skimage import  io,measure
import cv2
import  os
import numpy as np
import tifffile
import shutil
import math

def Illumination_select(Illu_path,image_root,Save_root,thres):
    """
        Select image for network training and testing by 3D mean Intensity.
        If <= threshold, the most part of the image is background.
    """
    if not os.path.exists(Save_root):
        os.mkdir(Save_root)
    image_list = os.listdir(image_root)
    with open(Illu_path,'r') as f:
        txt_lines = f.readlines()
    for i in range(len(txt_lines)):
        image_name = image_list[i]
        line = txt_lines[i]
        inten_list = list(map(float,line[1:-2].split(', ')))
        inten_3D = np.mean(inten_list)
        print(inten_3D)
        image_txt = os.path.join(Save_root,'image_path.txt')
        if inten_3D > thres:
            print(i, image_name)
            image_path = os.path.join(image_root,image_name)
            save_path = os.path.join(Save_root,image_name)
            shutil.copyfile(image_path,save_path)
            with open(image_txt,'a+') as f:
                f.writelines(image_name+'\t'+str(inten_3D)+'\n')


def Trainset_norm(Image_path, Label_path, Trainset_root):
    """
        Generate Training set from Image_path and Label_path.
        Dir Tree:
        ---Trainset_root\(Dataset_name\set1)
            ---image\
                ---00000\
                    00000.tif
                ...
            ---label\
                00000.txt
                ...
            train.txt
    """
    if not os.path.exists(Trainset_root):
        os.mkdir(Trainset_root)
    Trainset_image_root,Trainset_label_root = os.path.join(Trainset_root, 'image'), os.path.join(Trainset_root, 'label')

    # Mkdir as set1\image, set1\label
    if not os.path.exists(Trainset_image_root):
        os.mkdir(Trainset_image_root)
    if not os.path.exists(Trainset_label_root):
        os.mkdir(Trainset_label_root)
    # Generate data path and copy file.
    imagepath_list,labelpath_list = os.listdir(Image_path), os.listdir(Label_path)
    f = open(os.path.join(Trainset_root, 'train.txt'),'w+')
    for i in range(len(labelpath_list)):
        # Copy image file to training_set save path set1\image.
        imagepathi = os.path.join(Image_path, imagepath_list[i])
        save_image_dir,save_image_name = str(i).zfill(5), str(i).zfill(5) + '.tif'
        save_imagediri = os.path.join(Trainset_image_root, save_image_dir)
        save_imagepathi = os.path.join(save_imagediri, save_image_name)
        if not os.path.exists(save_imagediri):
            os.mkdir(save_imagediri)
        shutil.copyfile(imagepathi, save_imagepathi)
        # Copy label file to training_set save path set\label.
        labelpathi = os.path.join(Label_path, labelpath_list[i])
        save_label_name = str(i).zfill(5) + '.txt'
        save_labelpathi = os.path.join(Trainset_label_root, save_label_name)
        shutil.copyfile(labelpathi, save_labelpathi)
        print(imagepathi, '--->', save_imagepathi, '\n', labelpathi, '--->', save_labelpathi)
        # Write "train.txt"
        f.writelines(str(i).zfill(5)+'\n')
    f.close()

def Bouding_box_statics(Image_root,Label_root,thres_save_root,threshold):
    """
        Just for bounding box statics test , bounding box from .txt file.
    """
    num = 0
    count = np.zeros((10000,1))
    Image_list,Label_list = os.listdir(Image_root),os.listdir(Label_root)
    for i in range(len(Label_list)):
        image_path, bbox_path = os.path.join(Image_root, Image_list[i]),os.path.join(Label_root,Label_list[i])
        thres_save_path = os.path.join(thres_save_root,Label_list[i])
        with open(bbox_path, 'r') as f:
            bbox = f.readlines()
        print(image_path)
        image = io.imread(image_path).astype('uint16')
        bbox_num = 0
        filter_num = 0
        f = open(thres_save_path, 'w+')
        for item in bbox[1:]:
            bbox_num += 1
            x, y, z, r_xy, r_z = int(item.split(' ')[0]), int(item.split(' ')[1]), int(item.split(' ')[2]), \
                                 int(item.split(' ')[3]), int(item.split(' ')[4])
            x1, x2, y1, y2, z1, z2 = np.maximum(x - r_xy, 0), np.minimum(x + r_xy, 255), np.maximum(y - r_xy, 0), \
                                     np.minimum(y + r_xy, 255), np.maximum(z - r_z, 0), np.minimum(z + r_z, 127)
            image_bbox = image[z1:z2, y1:y2, x1:x2]
            mean_bbox, max_bbox, min_bbox, = np.mean(image_bbox), np.max(image_bbox), np.min(image_bbox)
            bbox_thres = max_bbox - min_bbox
            count[bbox_thres] += 1
            print(bbox_thres)
            f.writelines(str(x)+' '+str(y)+' '+str(z)+' '+str(r_xy)+' '+ str(r_z)+' '+str(bbox_thres) + '\n')
        f.close()
    thres = ''
    # for i in range(len(count)):

    for i in range(threshold):
        if count[i] != 0:
            thres = thres + str(i) + ':' + str(count[i]) + ' '
    print('Intensity(thres:count):', thres, '\n')


def Bbox_exp(Label_root,offset = 1):
    """
        For Bounding box expansion to include much more background pixels.
        Insight, the training performance will get better if the bounding box more fitting to objects.
    """
    Save_root = Label_root + '_offset_' + str(offset)
    if not os.path.exists(Save_root):
        os.mkdir(Save_root)
    Label_list = os.listdir(Label_root)
    for i in range(len(Label_list)):
        save_bbox_path = os.path.join(Save_root, Label_list[i])
        bbox_path = os.path.join(Label_root,Label_list[i])
        with open(bbox_path, 'r') as f:
            bbox = f.readlines()
        bbox_num = 0
        filter_num = 0
        pos = []
        for item in bbox[1:]:
            bbox_num += 1
            x, y, z , r_xy, r_z = int(item.split(' ')[0]), int(item.split(' ')[1]), int(item.split(' ')[2]), \
                    int(item.split(' ')[3]), int(item.split(' ')[4])
            x1, x2, y1, y2, z1, z2 = np.maximum(x - r_xy, 0), np.minimum(x + r_xy, 255), np.maximum(y - r_xy, 0), \
                                     np.minimum(y + r_xy, 255), np.maximum(z - r_z, 0), np.minimum(z + r_z, 74)
            '''
                To-do: Different offset with different bbox area. 
            '''
            x1_n, x2_n, y1_n, y2_n = np.maximum(x - r_xy - offset, 0), np.minimum(x + r_xy + offset, 255), \
                             np.maximum(y - r_xy - offset, 0), np.minimum(y + r_xy + offset, 255)
            x_n, y_n, r_xy_n = math.floor((x1_n+x2_n)/2), math.floor((y1_n+y2_n)/2),  math.floor((x2_n-x1_n)/2)
            print(x_n, y_n, z, r_xy_n, r_z)
            pos.append(str(x_n) + ' ' + str(y_n) + ' ' + str(z) + ' ' + str(r_xy_n) + ' ' + str(r_z) + '\n')
        with open(save_bbox_path, 'w+') as f:
            f.writelines("position_x position_y position_z radius_xy radius_z\n")
            for i in range(len(pos)):
                f.writelines(pos[i])

def draw_bbox_from_label(Label_root, Display_im_root, Display_save_root):
    """
        This method can used to draw the bounding boxes on the image for visual inspection.
        Label_root: ‘。txt’ file of labels.
        Display_im_root: '.tif' file of raw images.
        Display_save_root: Saved images with bounding boxes.
    """
    if not os.path.exists(Display_save_root):
        os.mkdir(Display_save_root)
    #Display_im_list = os.listdir(Display_im_root)
    Label_list = os.listdir(Label_root)
    for i in range(len(Label_list)):
        id = 0
        Label_path = os.path.join(Label_root, Label_list[i])
        Display_im_path = os.path.join(Display_im_root, Label_list[i].split('.')[0]+'.tif')
        image = io.imread(Display_im_path).astype('uint16')
        new_image = image
        Display_save_path = os.path.join(Display_save_root, Label_list[i].split('.')[0]+'.tif')
        print(Label_path,Display_im_path)
        with open(Label_path,'r') as f:
            bbox = f.readlines()
        for item in bbox[1:]:
            id += 1
            x, y, z, r_xy, r_z = int(item.split(' ')[0]), int(item.split(' ')[1]), int(item.split(' ')[2]), \
                                 int(item.split(' ')[3]), int(item.split(' ')[4])
            x1, x2, y1, y2, z1, z2 = np.maximum(x - r_xy, 0), np.minimum(x + r_xy, 255), np.maximum(y - r_xy, 0), \
                                     np.minimum(y + r_xy, 255), np.maximum(z - r_z, 0), np.minimum(z + r_z, 127)
            print(x1, x2, y1, y2, z1, z2)

            for z in range(z1, z2+1):
                cv2.rectangle(new_image[z, :, :], (x1, y1), (x2, y2), (0, 0xFFFF, 0), thickness=1)
                cv2.putText(new_image[z, :, :], str(id), (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.3,\
                            color=(0, 0xFFFF, 0), thickness=1)
            io.imsave(Display_save_path, (new_image))


def Radius_statics(Label_root):
    """
        Count the label's radius.
        1)Noisy labels can be filtered by irrational size.
        2)Select the RPN size by count result.
    """
    Hash_xy,Hash_z = np.zeros((100,1)),np.zeros((100,1))
    Label_list = os.listdir(Label_root)
    count = 0
    for i in range(len(Label_list)):
        #print(Label_list[i])
        Label_path = os.path.join(Label_root,Label_list[i])
        with open(Label_path,'r') as f:
            bboxes = f.readlines()
        for item in bboxes[1:]:
            r_xy, r_z = int(item.split(' ')[3]), int(item.split(' ')[4])
            Hash_xy[r_xy] += 1
            Hash_z[r_z] += 1
            count += 1
    xy = z = ''
    for i in range(len(Hash_xy)):
        if Hash_xy[i] != 0:
            xy = xy + str(i) + ':' +str(Hash_xy[i]) +' '
        if Hash_z[i] != 0:
            z = z + str(i) + ':' +str(Hash_z[i]) + ' '
    print('r_xy(len:count):',xy,'\n','r_z(size:count)',z, '\nBBOX_NUM:',count)


def Label_select(Label_root,Image_root,Save_root,size_thre=None,Signal_thre=None):
    """
        Select the labels with two filters.
        1) Size filter: Filter the incorrect labels with the size threshold.
            size_thre = [xy_low, xy_high, z_low, z_high]
        2) Signal filter: Filter the incorrect labels with the signal defference threshold.
            (max intensity - min intensity) within the bounding box.
    """
    if not os.path.exists(Save_root):
        os.mkdir(Save_root)
    Label_list = os.listdir(Label_root)
    for i in range(len(Label_list)):
        Label_path = os.path.join(Label_root, Label_list[i])
        image_path = os.path.join(Image_root, Label_list[i].split('.')[0] + '.tif')
        save_bbox_path = os.path.join(Save_root, Label_list[i])
        with open(Label_path, 'r') as f:
            bboxes = f.readlines()
        pos_new = []
        for item in bboxes[1:]:
            x, y, z, r_xy, r_z = int(item.split(' ')[0]), int(item.split(' ')[1]), int(item.split(' ')[2]), \
                                 int(item.split(' ')[3]), int(item.split(' ')[4])
            # Size filter
            print(x, y, z, r_xy, r_z)
            if size_thre is not None:
                xy_low, xy_high, z_low, z_high = size_thre[0], size_thre[1], size_thre[2], size_thre[3]
                if r_xy<=xy_low or r_xy>=xy_high or r_z<=z_low or r_z>=z_high:
                    continue
            # Signal filter
            if Signal_thre is not None:
                image = io.imread(image_path).astype('uint16')
                x1, x2, y1, y2, z1, z2 = np.maximum(x - r_xy, 0), np.minimum(x + r_xy, 255), np.maximum(y - r_xy, 0), \
                                         np.minimum(y + r_xy, 255), np.maximum(z - r_z, 0), np.minimum(z + r_z, 127)
                image_bbox = image[z1:z2, y1:y2, x1:x2]
                print(x1, x2, y1, y2, z1, z2)
                mean_bbox, max_bbox, min_bbox, = np.mean(image_bbox), np.max(image_bbox), np.min(image_bbox)
                if max_bbox - min_bbox < Signal_thre:
                    continue
            pos_new.append(str(x) + ' ' + str(y) + ' ' + str(z) + ' ' + str(r_xy) + ' ' + str(r_z) + '\n')
        with open(save_bbox_path, 'w+') as f:
            f.writelines("position_x position_y position_z radius_xy radius_z\n")
            for i in range(len(pos_new)):
                f.writelines(pos_new[i])

def BrainImage_reshape(BrainImage_root,Save_reshape_root,scale):
    """
        Reshape the brain image in z-axis, as for the network training need a standard size.
    """
    from scipy.ndimage import zoom
    if not os.path.exists(Save_reshape_root):
        os.mkdir(Save_reshape_root)
    BrainImage_list = os.listdir(BrainImage_root)
    for i in range(len(BrainImage_list)):
        BrainImge_path = os.path.join(BrainImage_root,BrainImage_list[i])
        Save_reshape_path = os.path.join(Save_reshape_root,BrainImage_list[i])
        print(BrainImge_path)
        image = tifffile.imread(BrainImge_path)
        print(image.shape)
        resized_data = zoom(image,(scale,1,1))
        print(resized_data.shape)
        tifffile.imwrite(Save_reshape_path,resized_data.astype(('uint16')),compress=2)


def Testset_norm(Image_path, Testset_root,Linux_root,stride=1):
    """
           Generate test set from Image_path.
           Dir Tree:
           ---Testset_root\(Dataset_name\set2)
               ---image\
                   ---00000\
                       00000.tif
                   ...
               ---label\
                   00000.txt
                   ...
               test.txt
               imagepath.txt
               index.txt
       """
    if not os.path.exists(Testset_root):
        os.mkdir(Testset_root)
    Testset_image_root = os.path.join(Testset_root, 'image')
    # Mkdir as set1\image, set1\label
    if not os.path.exists(Testset_image_root):
        os.mkdir(Testset_image_root)
    # Generate data path and copy file.
    imagepath_list = os.listdir(Image_path)
    f = open(os.path.join(Testset_root, 'test.txt'),'w+')
    f2 = open(os.path.join(Testset_root, 'imagepath.txt'),'w+')
    f3 = open(os.path.join(Testset_root,'index.txt'),'w+')
    for i in range(0,len(imagepath_list),stride):
        print(imagepath_list[i])
        # Copy image file to training_set save path set1\image.
        imagepathi = os.path.join(Image_path, imagepath_list[i])
        save_image_dir,save_image_name = str(i).zfill(5), str(i).zfill(5) + '.tif'
        save_imagediri = os.path.join(Testset_image_root, save_image_dir)
        save_imagepathi = os.path.join(save_imagediri, save_image_name)
        if not os.path.exists(save_imagediri):
            os.mkdir(save_imagediri)
        shutil.copyfile(imagepathi, save_imagepathi)
        print(imagepathi, '--->', save_imagepathi, '\n')
        # Write "train.txt"
        f.writelines(str(i).zfill(5)+'\n')
        f2.writelines(Linux_root + '/' + str(i).zfill(5) + '/' + str(i).zfill(5) + '.tif\n')
        f3.writelines(str(i).zfill(5) + ' ' + imagepath_list[i] + '\n')
    f.close()
    f2.close()

def Write_labelpath(Testset_root,Linux_root, test_num):
    """
        Write Imagepath.txt with Linux root.
    """
    imagepath = os.path.join(Testset_root, 'imagepath.txt')
    f = open(imagepath,'w+')
    for i in range(test_num):
        f.writelines(Linux_root + '/' + str(i).zfill(5) + '/' + str(i).zfill(5)+'.tif\n')
    f.close()


if __name__ == '__main__':
    pass