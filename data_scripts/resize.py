import torchvision
import os
from PIL import Image
import sys

base_dir = sys.argv[1]
new_dir = sys.argv[2]

resize_transform = torchvision.transforms.Resize(72)

assert(not os.path.exists(new_dir))

for split in ['train', 'val', 'test']:
    split_dir = base_dir + '/' + split
    classes = os.listdir(split_dir)
    for clas in classes:
        class_dir = split_dir + '/' + clas
        new_class_dir = class_dir.replace(base_dir, new_dir)
        os.makedirs(new_class_dir, exist_ok=True)
        for im_name in os.listdir(class_dir):
            im_path = class_dir + '/' + im_name
            new_im_path = im_path.replace(base_dir, new_dir)
            print(im_path)
            im = Image.open(im_path)
            im = resize_transform(im)
            if im.mode != 'RGB': im = im.convert('RGB')
            im.save(new_im_path)
