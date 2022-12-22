import  os
import numpy as np
import tifffile
import math



def Montage(Index_path,Binary_root,Save_root,MODE='3M'):
    """
    For 3M.
    Montage image from binary dir with index.txt generated during norm testset stage.
    Need to rewrite.
    :param Index_path:
    :param Binary_root:
    :param Save_root:
    :return:
    """
    if not os.path.exists(Save_root):
        os.mkdir(Save_root)
    with open(Index_path,'r') as f:
        lines = f.readlines()
    if MODE == '3M':
        # row = 11, column = 16, [2816, 4096]
        c = 16
        r = 11
        image_num = 85
        block_size = 256
        z_len = 128

    image_index_mat = np.full((image_num+1,c*r),-1)
    print(c,r,image_num)
    for line in lines:
        line = line.strip().split(' ')
        binary_image_index,image_realname = line[0],line[1]
        name_split = image_realname.split('.')[0].split('_')
        realimage_index, realimage_split_num = int(name_split[3]),int(name_split[7])
        image_index_mat[realimage_index, realimage_split_num] = binary_image_index

    c_index = np.argwhere(image_index_mat[:] != -1)
    image_list = np.unique(c_index[:, 0])

    for i in range(len(image_list)):
        Save_path = os.path.join(Save_root, str(image_list[i]).zfill(5)+'.tif')
        print(image_list[i])
        image = np.zeros((z_len,block_size*r,block_size*c)).astype('uint16')
        insert_pos = np.argwhere(image_index_mat[image_list[i]]!=-1)
        for j in range(len(insert_pos)):
            Block_name = str(int(image_index_mat[image_list[i], insert_pos[j]])).zfill(5)+'.tif'
            Block_path = os.path.join(Binary_root, Block_name)
            print(Block_path)
            block_image = tifffile.imread(Block_path)
            block_num = int(insert_pos[j])
            row,column= math.floor(block_num/c), block_num % c
            print(block_num,row,column)
            image[:, row*block_size:row*block_size+block_size ,column*block_size\
                    :column*block_size+block_size] = block_image
        tifffile.imwrite(Save_path,image,compress = 2)

if __name__ == '__main__':
    Index_path = ''
    Binary_root = ''
    Save_root = ''
    Montage(Index_path,Binary_root,Save_root)