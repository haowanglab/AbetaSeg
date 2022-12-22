import os
import tifffile
import csv
from skimage import measure
from math import ceil,floor
import numpy as np
import shutil

def BrainImage_cat(BrainImage_root,Save_root,flip=True, slices=75):
    """
        Concatenate the brain image slices to the 300 um image stack.
    """
    x = 4500
    y = 6300
    if not os.path.exists(Save_root):
        os.mkdir(Save_root)
    brainimage_list =  os.listdir(BrainImage_root)
    for i in range(floor(len(brainimage_list)/slices)):
        slice_list = brainimage_list[slices*i:slices*(i+1)]
        slice_image = np.zeros((slices,x,y)).astype('uint16')
        for j in range(len(slice_list)):
            print(j, os.path.join(BrainImage_root,slice_list[j]))
            im = tifffile.imread(os.path.join(BrainImage_root,slice_list[j]))
            if flip:
                im = np.fliplr(im)
            slice_image[j,: ,:] = im
        Save_path = os.path.join(Save_root,str(i).zfill(5)+'.tif')
        tifffile.imwrite(Save_path,slice_image,compress=2)


def BrainImage2Spot(BrainImage_root,csv_root1):
    """
        Get the plaque localization, morphology in each brain slice.
        The result saved in 'total.txt'.
    """
    if not os.path.exists(csv_root1):
        os.mkdir(csv_root1)
    brainimage_list = os.listdir(BrainImage_root)
    save_file = open(os.path.join(csv_root1,'total.txt'),'w+')
    save_file.writelines('Id X Y Z Area\n')
    for i in range(0,len(brainimage_list)):
        brainimage_path = os.path.join(BrainImage_root,brainimage_list[i])
        print(brainimage_path)
        print('image reading...')
        binary_image = tifffile.imread(brainimage_path)
        labeled_img = measure.label(binary_image, connectivity=1)
        properties = measure.regionprops(labeled_img)
        centroid_list = []
        area_list = []
        print('cell counting...')
        for pro in properties:
            centroid = pro.centroid
            centroid_list.append(centroid)
            area = pro.area
            area_list.append(area)
        centroid_list.sort()
        for j in range(len(centroid_list)):
            z = ceil(centroid_list[j][0])
            y = ceil(centroid_list[j][1])
            x = ceil(centroid_list[j][2])
            area = area_list[j]
            z_index = z + i*75
            print(x, y, z_index, area, '---', j)
            content = str(j) + ' ' + str(x) + ' '+ str(y) +' ' + str(z_index) +' ' + str(area) + '\n'
            save_file.writelines(content)
    save_file.close()



def Spot_csv(total_path, csv_root):
    """
        Tranform the 'total.txt' into Thumbnail's csv file, which is needed as input of Freesia2.
    """
    if not os.path.exists(csv_root):
        os.mkdir(csv_root)
    f = open(total_path,'r')
    spots = []
    for spot in f.readlines()[1:]:
        a = spot.strip().split(' ')
        a = np.array(a).astype(dtype=int).tolist()
        spots.append(a)
    print(spots)

    # group with z = spots[3] , for every 9 slices
    for i in range(floor(6375/9)):
        print('----Thumbnail'+str(i))
        csv_name = str(i).zfill(4)+'_36.0.tif.csv'
        csv_path = os.path.join(csv_root,csv_name)
        count = 0
        list_name = ['Position X', 'Position Y', 'Position Z', 'Unit', 'Category', 'Collection','Time','ID']
        with open(csv_path,'w+',newline='')as f:
            csv_write = csv.writer(f, dialect='excel')
            csv_write.writerow(['36.0'])
            csv_write.writerow(['=========='])
            csv_write.writerow(list_name)
            for j in range(len(spots)):
                if i*9<=spots[j][3]<=i*9+8:
                    print(spots[j])
                    x ,y ,z = str(spots[j][1]*4), str(spots[j][2]*4), str(0)
                    writeline = [x,y,z,'um','Spot','Position','1',str(count)]
                    count += 1
                    csv_write.writerow(writeline)

def flip(Stack_root, flip_save_root):
    if not os.path.exists(flip_save_root):
        os.mkdir(flip_save_root)
    stack_list = os.listdir(Stack_root)
    for i in range(len(stack_list)):
        pass

def Group_reform(Group_root, Save_root):
    """
        A tool for export group by group.
    """
    if not os.path.exists(Save_root):
        os.mkdir(Save_root)
    save_im_root = os.path.join(Save_root, 'images')
    if not os.path.exists(save_im_root):
        os.mkdir(save_im_root)
    group = os.listdir(Group_root)
    count = 0
    for i in range(len(group)):
        group_path = os.path.join(Group_root,str(i+1))
        print(group_path)
        group_image_root = os.path.join(group_path, 'images')
        # print(group_image_root)
        for j in range(8):
            im_name = str(count).zfill(4)+'_36.tif'
            json_name = str(count).zfill(4)+'_36.json'
            im_path = os.path.join(group_image_root,im_name)
            json_path = os.path.join(group_path, json_name)
            save_im_path = os.path.join(save_im_root,im_name)
            save_json_path = os.path.join(Save_root, json_name)
            print(im_path,'->',save_im_path,'\n',json_path,'->',save_json_path)
            shutil.copyfile(im_path, save_im_path)
            shutil.copyfile(json_path, save_json_path)
            count += 1
            if count == 708:
                break

if __name__ == '__main__':

    # # 3M scripts
    # 1) After segmentation postprocessing and automatic registration.
    BrainImage_root = ''
    Save_root = ''
    BrainImage_cat(BrainImage_root, Save_root, flip=True)

    # 2 Before import to freesia2.
    BrainImage_root = Save_root
    csv_root1 = ''
    BrainImage2Spot(BrainImage_root, csv_root1)
    csv_root = ''
    total_path = ''
    Spot_csv(total_path, csv_root)

    # 3 After export from freesia2.
    Group_root = ''
    Save_root = ''
    Group_reform(Group_root, Save_root)