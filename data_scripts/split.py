import shutil
import csv
import random
import os
import sys

image_dir = sys.argv[1]
categories = os.listdir(image_dir)
for split in ['train', 'test', 'val']:
    if split in categories:
        categories.remove(split)
        shutil.rmtree(image_dir + split)
    for category in categories:
        os.makedirs(image_dir + split + '/' + category, exist_ok=True)

for category in categories:
    cat_dir = '/'.join([image_dir, category])
    all_images = os.listdir(cat_dir)
    random.shuffle(all_images)
    num_images = len(all_images)
    train_val_cutoff = num_images * 0.6
    val_test_cutoff = num_images * 0.8
    for i, image in enumerate(all_images):
        image_path = '/'.join([cat_dir, image])
        print(i, train_val_cutoff, val_test_cutoff)
        if i < train_val_cutoff:
            split_dir = 'train/'
        elif i < val_test_cutoff:
            split_dir = 'val/'
        else:
            split_dir = 'test/'
        print(image_path, '/'.join([image_dir, split_dir, category]))
        shutil.copy(image_path, '/'.join([image_dir, split_dir, category]))



