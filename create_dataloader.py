import os
import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision import transforms, models
import torchvision.transforms.functional as TF
from torchvision.transforms import Compose, CenterCrop, Normalize, ToTensor
# I have placed a manual seed in line 34 to make code reproducible

def get_data(flair_imgs_dir, t1ce_imgs_dir, t2_imgs_dir, gt_imgs_dir, train_val_test_split=(0.7, 0.15, 0.15)):
    # First we get the file paths for all the input images and the corresponding ground truth images
    flair_imgs_list = os.listdir(flair_imgs_dir)
    t1ce_imgs_list = os.listdir(t1ce_imgs_dir)
    t2_imgs_list = os.listdir(t2_imgs_dir)
    gt_imgs_list = os.listdir(gt_imgs_dir)
    # print(len(flair_imgs_list))
    # print(len(t1ce_imgs_list))
    # print(len(t2_imgs_list))
    # print(len(gt_imgs_list))
    flair_imgs_list.sort()
    t1ce_imgs_list.sort()
    t2_imgs_list.sort()
    gt_imgs_list.sort()

    # We want the image lists to contain file paths (not file names)
    for i in range(len(flair_imgs_list)):
        flair_imgs_list[i] = os.path.join(flair_imgs_dir, flair_imgs_list[i])
        t1ce_imgs_list[i] = os.path.join(t1ce_imgs_dir, t1ce_imgs_list[i])
        t2_imgs_list[i] = os.path.join(t2_imgs_dir, t2_imgs_list[i])
        gt_imgs_list[i] = os.path.join(gt_imgs_dir, gt_imgs_list[i])

    # Check if number of images in all lists are the same
    flag = len(np.unique(np.array([len(flair_imgs_list), len(t1ce_imgs_list), len(t2_imgs_list), len(gt_imgs_list)])))
    assert flag == 1, "The number of images in different inputted folders vary!"

    # Computing the length of the train validation and test subsets
    trn_subset_len = int((train_val_test_split[0]) * len(flair_imgs_list))
    vld_subset_len = int((len(flair_imgs_list) - trn_subset_len) * (
                (train_val_test_split[1]) / (train_val_test_split[1] + train_val_test_split[2])))
    tst_subset_len = len(flair_imgs_list) - trn_subset_len - vld_subset_len

    # Shuffle the lists (all dir lists get shuffled the same way)
    np.random.seed(42)
    p = np.random.permutation(len(flair_imgs_list))
    flair_imgs_list = ((np.array(flair_imgs_list))[p]).tolist()
    t1ce_imgs_list = ((np.array(t1ce_imgs_list))[p]).tolist()
    t2_imgs_list = ((np.array(t2_imgs_list))[p]).tolist()
    gt_imgs_list = ((np.array(gt_imgs_list))[p]).tolist()

    # Perform train, validation and test split on data
    flair_trn = flair_imgs_list[:trn_subset_len]
    flair_vld = flair_imgs_list[trn_subset_len: trn_subset_len + vld_subset_len]
    flair_tst = flair_imgs_list[-tst_subset_len:]

    t1ce_trn = t1ce_imgs_list[:trn_subset_len]
    t1ce_vld = t1ce_imgs_list[trn_subset_len: trn_subset_len + vld_subset_len]
    t1ce_tst = t1ce_imgs_list[-tst_subset_len:]

    t2_trn = t2_imgs_list[:trn_subset_len]
    t2_vld = t2_imgs_list[trn_subset_len: trn_subset_len + vld_subset_len]
    t2_tst = t2_imgs_list[-tst_subset_len:]

    gt_trn = gt_imgs_list[:trn_subset_len]
    gt_vld = gt_imgs_list[trn_subset_len: trn_subset_len + vld_subset_len]
    gt_tst = gt_imgs_list[-tst_subset_len:]

    return (flair_trn, t1ce_trn, t2_trn, gt_trn), (flair_vld, t1ce_vld, t2_vld, gt_vld), (
    flair_tst, t1ce_tst, t2_tst, gt_tst)


class Train_Dataset(Dataset):
    '''
    Custom dataset defined for our BraTs Data for training
    The data has been randomly cropped to 160*160
    '''

    def __init__(self, flair_imgs_list, t1ce_imgs_list, t2_imgs_list, gt_imgs_list):
        '''
        flair_imgs_list: a list containing all directories for "flair" modality images in the train data
        t1ce_imgs_list: a list containing all directories for "t1ce" modality images in the train data
        t2_imgs_list: a list containing all directories for "t2" modality images in the train data
        gt_imgs_list: a list containing all directories for "seg" modality images in the train data  -> segmented images

        '''
        self.flair_imgs_list = flair_imgs_list
        self.t1ce_imgs_list = t1ce_imgs_list
        self.t2_imgs_list = t2_imgs_list
        self.gt_imgs_list = gt_imgs_list

    def transform(self, flair_PIL, t1ce_PIL, t2_PIL, gt_PIL):
        '''
        Performs identical random transformation on flair, t1ce, t2 and gt images

        Returns:
        (input_img, gt_img) --> all sizes: (160*160)
        '''
        # Get random crop coordinates
        torch.manual_seed(42)
        top, left, height, width = transforms.RandomCrop.get_params(flair_PIL, (160, 160))
        # Pass the coordinates to F.crop --> Results in 160*160 numpy arrays
        # flair_np = np.array(TF.crop(flair_PIL, top, left, height, width))
        # t1ce_np = np.array(TF.crop(t1ce_PIL, top, left, height, width))
        # t2_np = np.array(TF.crop(t2_PIL, top, left, height, width))
        # gt_np = np.array(TF.crop(gt_PIL, top, left, height, width))

        flair_np = np.array(TF.resize(flair_PIL,(160,160)))
        t1ce_np = np.array(TF.resize(t1ce_PIL,(160,160)))
        t2_np = np.array(TF.resize(t2_PIL,(160,160)))
        gt_np=np.array(TF.resize(gt_PIL,(160,160)))

        img_transform_3 = Compose([
            ToTensor(),
            Normalize(mean=[0.5,0.5,0.5], std=[0.5,0.5,0.5]),
        ])
        img_transform_1 = Compose([
            ToTensor(),
            Normalize(mean=[0.5], std=[0.5]),
        ])
        flair_t=img_transform_1(flair_np)
        t1ce_t = img_transform_1(t1ce_np)
        t2_t = img_transform_1(t2_np)
        gt_t = img_transform_3(gt_np)
        # Create ground_truth mask
        # gt_3d = np.zeros((3, 160, 160))
        # gt_3d[0, :, :] = (gt_np == 1)
        # gt_3d[1, :, :] = (gt_np == 2)
        # gt_3d[2, :, :] = (gt_np == 4)
        # gt_3d[0, :, :] =gt_np[:,:,0]
        # gt_3d[1, :, :] =gt_np[:,:,1]
        # gt_3d[2, :, :] =gt_np[:,:,2]
        #gt_3d=gt_np.transpose(2,0,1)
        #gt_3d_t=gt_t.transpose(2,0,1)
        # gt_3d_t=torch.transpose(gt_t,1,2)
        # gt_3d_t = torch.transpose(gt_3d_t, 0, 1)
        input_img_t=torch.stack((flair_t, t1ce_t, t2_t), axis=0)
        input_img_t=torch.squeeze(input_img_t)

        # Stack the input images
        #input_img = np.stack((flair_np, t1ce_np, t2_np), axis=0)


        # Return input as a C*H*W tensor and ground_truth mask as  3*H*W tensor -> 3 is for the 3 types of tumors
        #return (torch.tensor(input_img, dtype=torch.float)), (torch.tensor(gt_3d, dtype=torch.float))
        return (torch.tensor(input_img_t, dtype=torch.float)), (torch.tensor(gt_t, dtype=torch.float))

    def __len__(self):
        '''returns the length of the entire dataset'''
        return len(self.flair_imgs_list)

    def __getitem__(self, idx):
        # Open Images
        flair_PIL = Image.open(self.flair_imgs_list[idx])
        t1ce_PIL = Image.open(self.t1ce_imgs_list[idx])
        t2_PIL = Image.open(self.t2_imgs_list[idx])
        gt_PIL = Image.open(self.gt_imgs_list[idx])

        # Perform Transformation
        input_img, gt_3d = self.transform(flair_PIL=flair_PIL, t1ce_PIL=t1ce_PIL, t2_PIL=t2_PIL, gt_PIL=gt_PIL)

        return input_img, gt_3d


class vld_tst_Dataset(Dataset):
    '''
    Custom dataset defined for our BraTs Data for validation and testing
    The data has been center cropped to 160*160
    '''

    def __init__(self, flair_imgs_list, t1ce_imgs_list, t2_imgs_list, gt_imgs_list):
        '''
        flair_imgs_list: a list containing all directories for "flair" modality images in the validation and test data
        t1ce_imgs_list: a list containing all directories for "t1ce" modality images in the validation and test data
        t2_imgs_list: a list containing all directories for "t2" modality images in the validation and test data
        gt_imgs_list: a list containing all directories for "seg" modality images in the validation and test data  -> segmented images

        '''
        self.flair_imgs_list = flair_imgs_list
        self.t1ce_imgs_list = t1ce_imgs_list
        self.t2_imgs_list = t2_imgs_list
        self.gt_imgs_list = gt_imgs_list
        self.center_crop = transforms.Compose([
            transforms.CenterCrop(160)
        ])

    def transform(self, flair_PIL, t1ce_PIL, t2_PIL, gt_PIL):
        '''
        Performs required transformations on images
        '''
        flair_np = np.array(self.center_crop(flair_PIL))
        t1ce_np = np.array(self.center_crop(t1ce_PIL))
        t2_np = np.array(self.center_crop(t2_PIL))
        gt_np = np.array(self.center_crop(gt_PIL))

        # Create ground_truth mask
        gt_3d = np.zeros((3, 160, 160))
        gt_3d[0, :, :] = (gt_np == 1)
        gt_3d[1, :, :] = (gt_np == 2)
        gt_3d[2, :, :] = (gt_np == 4)

        # Stack input images together
        input_img = np.stack((flair_np, t1ce_np, t2_np), axis=0)

        # Return input as a C*H*W tensor and ground_truth mask as  3*H*W tensor --> 3 is for the 3 types of tumors
        return torch.tensor(input_img, dtype=torch.float), torch.tensor(gt_3d, dtype=torch.float)

    def __len__(self):
        '''returns the length of the entire dataset'''
        return len(self.flair_imgs_list)

    def __getitem__(self, idx):
        # Open Images
        flair_PIL = Image.open(self.flair_imgs_list[idx])
        t1ce_PIL = Image.open(self.t1ce_imgs_list[idx])
        t2_PIL = Image.open(self.t2_imgs_list[idx])
        gt_PIL = Image.open(self.gt_imgs_list[idx])

        # Perform Transformation
        input_img, gt_3d = self.transform(flair_PIL=flair_PIL, t1ce_PIL=t1ce_PIL, t2_PIL=t2_PIL, gt_PIL=gt_PIL)

        return input_img, gt_3d