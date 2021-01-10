import torch.utils.data as data
import torchvision
import torchvision.transforms as transforms
import torch
from copy import copy
import numpy as np
import pickle
from PIL import Image
import os
import os.path
import random
from torchvision.datasets import ImageFolder

def __getitem__ImageFolder(self, index):
    path, target = self.samples[index]
    sample = self.loader(path)
    if self.transform is not None:
        sample = self.transform(sample)
    if self.target_transform is not None:
        target = self.target_transform(target)
    if self.yield_indices:
        return sample, target, index
    else:
        return sample, target

def shuffle_labels(self):
    paths, targets = zip(*self.samples)
    targets = list(targets)
    random.shuffle(targets)
    new_samples = list(zip(paths, targets))
    self.samples = new_samples

ImageFolder.__getitem__ = __getitem__ImageFolder
ImageFolder.shuffle_labels = shuffle_labels

def prepare_data_loaders(data_dir, shuffle_train=True, index=None,
                         train_batch_size=128, test_batch_size=100,
                         image_dim=64, resize_dim=72, test=False, yield_indices=False,
                         train_on_10_percent=False,
                         train_on_half_classes=False,
                         shuffle_val=False,
                         no_flip=False):
    num_classes = len(os.listdir(data_dir + '/train/'))
    train = [0]
    val = [1]

    means=[0.5, 0.5, 0.5]
    stds=[0.5, 0.5, 0.5]

    flip_transform = transforms.RandomHorizontalFlip()

    transform_train = transforms.Compose([
        transforms.Resize(resize_dim),
        transforms.RandomCrop(image_dim),
        flip_transform,
        transforms.ToTensor(),
        transforms.Normalize(means, stds),
        ])  
    transform_test = transforms.Compose([
        transforms.Resize(resize_dim),
        transforms.CenterCrop(image_dim),
        transforms.ToTensor(),
        transforms.Normalize(means, stds),
        ])
    ImageFolder.yield_indices = yield_indices

    if train_on_10_percent:
        train_dir_string = data_dir + '/train10/'
        val_dir_string = data_dir + '/val10/'
    elif train_on_half_classes:
        train_dir_string = data_dir + '/train_half_classes/'
        val_dir_string = data_dir + '/val_half_classes/'
    else:
        train_dir_string = data_dir + '/train/'
        val_dir_string = data_dir + '/val/'

    trainloader = torch.utils.data.DataLoader(ImageFolder(train_dir_string, transform_train),
                                                          batch_size=train_batch_size,
                                                          shuffle=shuffle_train, num_workers=4, pin_memory=True)
    if test:
        valloader = torch.utils.data.DataLoader(ImageFolder(data_dir + '/test/', transform_test),
                                                            batch_size=test_batch_size,
                                                            shuffle=False, num_workers=4, pin_memory=True)
    else:
        valloader = torch.utils.data.DataLoader(ImageFolder(val_dir_string, transform_test),
                                                            batch_size=test_batch_size,
                                                            shuffle=shuffle_val, num_workers=4, pin_memory=True)

    return trainloader, valloader, num_classes
