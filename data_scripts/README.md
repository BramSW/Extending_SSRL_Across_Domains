# split.py

This is the script used to generate data splits.

Say we have an image directory `dataset` with subdirectories containing classes `dataset/class1 dataset/class2 ...`.

Running `python split.py dataset` will create subdirectories `dataset/train dataset/val dataset/test`
each having the same subdirectory structure as the original `dataset` directory and containing a 60-20-20 per-class split of images.

# resize.py

This is the script used to resize images to 72x72 resolution (cropped to 64x64 as input)

From the above example, run `python3 resize.py dataset final_dataset`.

The train/val/test subdirectories will be copied into `final_dataset` with the images resized to 72x72.
