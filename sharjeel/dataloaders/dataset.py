import os
import cv2
import torch
import random
import numpy as np
from glob import glob
from torch.utils.data import Dataset
import h5py
from scipy.ndimage.interpolation import zoom
import itertools
from scipy import ndimage
from torch.utils.data.sampler import Sampler
from torchvision import transforms

class prepare_dataset(Dataset):
    def __init__(self, data_path, state = 'train'):
      self.data = np.load(data_path, allow_pickle = True)
      self.state = state
      self.transform = transforms.Compose([
        RandomGenerator([256, 256])
    ])
    
    def __len__(self):
      return len(self.data)
      
    def __getitem__(self, idx):
      image_path, mask_path = self.data[idx]
      #print(idx)
      
      image = cv2.imread(image_path, 0)
      mask = cv2.imread(mask_path, 0)
      #mask = np.where(mask > 1, 0, mask)
      #mask[mask == 255] = 1
      #mask[mask != 0] = 1
      mask[mask <= 220] = 0
      mask[mask != 0] = 255
      mask[mask == 255] = 1
      shape_x, shape_y = image.shape
      #print('image shape: ', image.shape)
      if shape_x != 256 or shape_y != 256:
        image = cv2.resize(image, (256, 256), interpolation=cv2.INTER_CUBIC) # for normal
        mask = cv2.resize(mask, (256, 256), interpolation=cv2.INTER_NEAREST)
        
        #image = cv2.resize(image, (128, 128), interpolation=cv2.INTER_CUBIC) # for patch
        #mask = cv2.resize(mask, (128, 128), interpolation=cv2.INTER_NEAREST)
      
      #print('image shape: ', image.shape)
      sample = {'image': image, 'label': mask}
      
      if self.state == 'train':
        sample = self.transform(sample)
        
      sample["idx"] = idx
      return sample

class prepare_dataset_crop(Dataset):
    def __init__(self, data_path, state='train'):
      self.data = np.load(data_path, allow_pickle = True)
      self.state = state
      self.transform = transforms.Compose([
        RandomGenerator([128, 128])
    ])
    
    def __len__(self):
      return len(self.data)
      
    def __getitem__(self, idx):
      image_path, mask_path = self.data[idx]
      #print(idx)
      
      image = cv2.imread(image_path, 0)
      mask = cv2.imread(mask_path, 0)
      #mask = np.where(mask > 1, 0, mask)
      #mask[mask == 255] = 1
      #mask[mask != 0] = 1
      mask[mask <= 220] = 0
      mask[mask != 0] = 255
      mask[mask == 255] = 1
      shape_x, shape_y = image.shape
      #print('image shape: ', image.shape)
      if shape_x != 128 or shape_y != 128:
        image = cv2.resize(image, (128, 128), interpolation=cv2.INTER_CUBIC) # for patch
        mask = cv2.resize(mask, (128, 128), interpolation=cv2.INTER_NEAREST)
      
      #print('image shape: ', image.shape)
      sample = {'image': image, 'label': mask}
      
      if self.state == 'train':
        sample = self.transform(sample)
        
      sample["idx"] = idx
      return sample

class BaseDataSets(Dataset):
    def __init__(self, base_dir=None, split='train', num=None, transform=None):
        self._base_dir = base_dir
        self.sample_list = []
        self.split = split
        self.transform = transform
        if self.split == 'train':
            with open(self._base_dir + '/train.list', 'r') as f1:  #change everytime for folds(comments: "train_fold_2.list, train_fold_3.list, train_fold_4.list, train_fold_5.list"
                self.sample_list = f1.readlines()
            self.sample_list = [item.replace('\n', '')
                                for item in self.sample_list]

        elif self.split == 'val':
#             with open("/home/centos/Farhan/pickle/ultrasound_2d/data/h5_ultrasound_3channel/val.list", "r") as f:
            with open(self._base_dir + '/val.list', 'r') as f: #change everytime for folds(comments: val_fold_2.list, val_fold_3.list, val_fold_4.list, val_fold_5.list)
                self.sample_list = f.readlines()
            self.sample_list = [item.replace('\n', '')
                                for item in self.sample_list]
        if num is not None and self.split == "train":
            self.sample_list = self.sample_list[:num]
        print("total {} samples".format(len(self.sample_list)))

    def __len__(self):
        return len(self.sample_list)

    def __getitem__(self, idx):
        case = self.sample_list[idx]
        if self.split == "train":
            h5f = h5py.File(self._base_dir +
                            "/{}".format(case), 'r')
        else:
#             h5f = h5py.File("/home/centos/Farhan/pickle/ultrasound_2d/data/h5_ultrasound_3channel" + "/{}".format(case), "r")
            h5f = h5py.File(self._base_dir + "/{}".format(case), 'r')
        image = h5f['image'][:]
        label = h5f['label'][:]
        sample = {'image': image, 'label': label}
        if self.split == "train":
            sample = self.transform(sample)
        sample["idx"] = idx
        return sample


def random_rot_flip(image, label):
    k = np.random.randint(0, 4)
    image = np.rot90(image, k)
    label = np.rot90(label, k)
    axis = np.random.randint(0, 2)
    image = np.flip(image, axis=axis).copy()
    label = np.flip(label, axis=axis).copy()
    return image, label


def random_rotate(image, label):
    angle = np.random.randint(-20, 20)
    image = ndimage.rotate(image, angle, order=0, reshape=False)
    label = ndimage.rotate(label, angle, order=0, reshape=False)
    return image, label


class RandomGenerator(object):
    def __init__(self, output_size):
        self.output_size = output_size

    def __call__(self, sample):
        image, label = sample['image'], sample['label']
        # ind = random.randrange(0, img.shape[0])
        # image = img[ind, ...]
        # label = lab[ind, ...]
        if random.random() > 0.5:
            image, label = random_rot_flip(image, label)
        elif random.random() > 0.5:
            image, label = random_rotate(image, label)
        x, y = image.shape
        image = zoom(
            image, (self.output_size[0] / x, self.output_size[1] / y), order=0)
        label = zoom(
            label, (self.output_size[0] / x, self.output_size[1] / y), order=0)
        image = torch.from_numpy(
            image.astype(np.float32)).unsqueeze(0)
        label = torch.from_numpy(label.astype(np.uint8))
        sample = {'image': image, 'label': label}
        return sample


class TwoStreamBatchSampler(Sampler):
    """Iterate two sets of indices

    An 'epoch' is one iteration through the primary indices.
    During the epoch, the secondary indices are iterated through
    as many times as needed.
    """

    def __init__(self, primary_indices, secondary_indices, batch_size, secondary_batch_size):
        self.primary_indices = primary_indices
        self.secondary_indices = secondary_indices
        self.secondary_batch_size = secondary_batch_size
        self.primary_batch_size = batch_size - secondary_batch_size

        assert len(self.primary_indices) >= self.primary_batch_size > 0
        assert len(self.secondary_indices) >= self.secondary_batch_size > 0

    def __iter__(self):
        primary_iter = iterate_once(self.primary_indices)
        secondary_iter = iterate_eternally(self.secondary_indices)
        return (
            primary_batch + secondary_batch
            for (primary_batch, secondary_batch)
            in zip(grouper(primary_iter, self.primary_batch_size),
                   grouper(secondary_iter, self.secondary_batch_size))
        )

    def __len__(self):
        return len(self.primary_indices) // self.primary_batch_size


def iterate_once(iterable):
    return np.random.permutation(iterable)


def iterate_eternally(indices):
    def infinite_shuffles():
        while True:
            yield np.random.permutation(indices)
    return itertools.chain.from_iterable(infinite_shuffles())


def grouper(iterable, n):
    "Collect data into fixed-length chunks or blocks"
    # grouper('ABCDEFG', 3) --> ABC DEF"
    args = [iter(iterable)] * n
    return zip(*args)
