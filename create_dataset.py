import gzip
import shutil
import os
import nibabel as nib
from tifffile import imsave, imread
import numpy as np
import matplotlib.pyplot as plt
import imageio
from array2gif import write_gif
from PIL import Image


def make_dir(path):
    if (not os.path.isdir(path)):
        os.makedirs(path)


image_types = ['flair', 'segmentation', 't1', 't1ce', 't2']
folder_name = 'MICCAI_BraTS2020_TrainingData'
dataset_path = 'dataset'

main_dir = os.listdir(folder_name)
make_dir(dataset_path)
for image_type in image_types:
    make_dir(os.path.join(dataset_path, image_type))

for folder in main_dir:
    cur_dir = os.path.join(folder_name, folder)
    cur_dir_files = os.listdir(cur_dir)
    for i, file_name in enumerate(cur_dir_files):
        #print(file_name)
        file_path = os.path.join(cur_dir, file_name)
        if 'seg' in file_name:
            # print(file)
            imgVol = nib.load(file_path)
            npdata = imgVol.get_fdata()
            npdata = npdata.astype(np.uint8)
            x = npdata.shape[0]
            y = npdata.shape[1]
            n = npdata.shape[2]
            array = np.zeros((x, y, 3, n))
            array[:, :, 0, :] = np.where(npdata == 1, 255 * np.ones((x, y, n)), np.zeros((x, y, n)))
            array[:, :, 1, :] = np.where(npdata == 2, 255 * np.ones((x, y, n)), np.zeros((x, y, n)))
            array[:, :, 2, :] = np.where(npdata == 4, 255 * np.ones((x, y, n)), np.zeros((x, y, n)))
            array = array.astype(np.uint8)
            array = array.transpose(3, 0, 1, 2)
            # print(array.shape)
            # plt.imshow(array[:,:,:,51])
            # write_gif(array.transpose((3,0,1,2)),f'movie{i}.gif',fps=300)
            save_path = os.path.join(dataset_path, 'segmentation', file_name)
            print('saving', file_name, 'in',save_path)
            for j, image in enumerate(array):
                # print(image.shape)
                im = Image.fromarray(image)
                im.save(save_path + str(j) + '.png')
        else:
            img_input_types=['flair',  't1ce', 't2','t1']
            for img_t in img_input_types:
                if img_t in file_name:
                    save_path = os.path.join(dataset_path, img_t, file_name)
                    #make_dir(os.path.join(dataset_path, folder, image_types[i]))
                    imgVol = nib.load(file_path)
                    npdata = imgVol.get_fdata()
                    npdata = npdata.astype(np.uint8)
                    for j, image in enumerate(npdata.transpose(2, 0, 1)):
                        im = Image.fromarray(image)
                        im.save(save_path + str(j) + '.png')









