import numpy as np
import os
import random
from os import path
from tqdm import tqdm
import matplotlib.image as img
import matplotlib.pyplot as plt
import imgaug as ia
from imgaug import augmenters as iaa
import glob
import re
import cv2



train_data = [1, 2, 3, 4] # data idx of train files to get images from(1.npz, 2.npz .... 10.npz)
test_data = [9] # data idx of test files
img_for_rotation = 40 #number of random picks for rotated augmentation
original_data_percentage = 0.1
overlap = True  #used for image generation from raw_data, if False ~40 unique images,
                    #else ~66 images with small overlap
rotational_augment_portion = 0.4 #portion of rotated images of all augmented images
test_overlap = False #overlap for test data


# numerical sorter for folder, for windows order
def numericalSort(value):
    numbers = re.compile(r'(\d+)')
    parts = numbers.split(value)
    parts[1::2] = map(int, parts[1::2])
    return parts

#generate folders for data images
if not path.exists('./data/depth/train/'):
    os.makedirs('./data/depth/train/')
if not path.exists('./data/target/train/'):
    os.makedirs('./data/target/train/')

if not path.exists('./test_data/depth/test/'):
    os.makedirs('./test_data/depth/test/')
if not path.exists('./test_data/target/test/'):
    os.makedirs('./test_data/target/test/')

#extract test images from raw data
print('loading test images...')
for idx in tqdm(test_data):
    data = np.load('{}.npz'.format(idx))
    depth = data['height']
    target = data['mask']

    depth = (depth - np.min(depth)) / np.ptp(depth)
    target = (target - np.min(target)) / np.ptp(target)

    plt.imsave('image_grain.png', depth)  # , cmap='viridis')
    depth = img.imread('image_grain.png')

    shape = np.shape(depth)
    img_per_row = int(shape[1] / 512)
    starts_y = [0, shape[0] - 512]

    if test_overlap:
        free_mid = shape[0] - 512 - 512
        offset = (512 - free_mid) / 2
        starts_y = [0, int(512 - offset), int(shape[0] - 512)] #top row, center row, bottom row
        if (shape[1] / 512) - img_per_row > 0.3:
            img_per_row += 1

    for j in range(len(starts_y)):
        for i in range(img_per_row):
            start_y = starts_y[j]
            if overlap and i==img_per_row-1:
                sub_depth = depth[start_y: start_y+512, shape[1] - 512: shape[1]]
                sub_target = target[start_y: start_y+512, shape[1] - 512: shape[1]]
            else:
                sub_depth = depth[start_y: start_y+512, 512 * i: 512*(i+1)]
                sub_target = target[start_y: start_y+512, 512 * i: 512*(i+1)]
            folder_g = './test_data/depth/test/'
            folder_m = './test_data/target/test/'
            plt.imsave(folder_g + 'image{:03d}_p{:d}{:d}.png'.format(idx, i, j), sub_depth, cmap='viridis')
            plt.imsave(folder_m + 'image{:03d}_p{:d}{:d}.png'.format(idx, i, j), sub_target, cmap='gray')


#extract train images from raw data
print('loading train images...')
image_counter = 0
image_counter_rot = 0
for idx in tqdm(train_data):
    data = np.load('{}.npz'.format(idx))
    depth = data['height']
    target = data['mask']

    depth = (depth - np.min(depth)) / np.ptp(depth)
    target = (target - np.min(target)) / np.ptp(target)

    plt.imsave('image_grain.png', depth)  # , cmap='viridis')
    depth = img.imread('image_grain.png')

    shape = np.shape(depth)
    img_per_row = int(shape[1] / 512)
    starts_y = [0, shape[0] - 512]

    if overlap:
        free_mid = shape[0] - 512 - 512
        offset = (512 - free_mid) / 2
        starts_y = [0, int(512 - offset), int(shape[0] - 512)] #top row, center row, bottom row
        if (shape[1] / 512) - img_per_row > 0.3:
            img_per_row += 1

    for j in range(len(starts_y)):
        for i in range(img_per_row):
            start_y = starts_y[j]
            if overlap and i==img_per_row-1:
                sub_depth = depth[start_y: start_y+512, shape[1] - 512: shape[1]]
                sub_target = target[start_y: start_y+512, shape[1] - 512: shape[1]]
            else:
                sub_depth = depth[start_y: start_y+512, 512 * i: 512*(i+1)]
                sub_target = target[start_y: start_y+512, 512 * i: 512*(i+1)]
            folder_g = './data/depth/train/'
            folder_m = './data/target/train/'
            plt.imsave(folder_g + 'image{:03d}_p{:d}{:d}.png'.format(idx, i, j), sub_depth, cmap='viridis')
            plt.imsave(folder_m + 'image{:03d}_p{:d}{:d}.png'.format(idx, i, j), sub_target, cmap='gray')
            image_counter += 1
    if overlap:
        #random picks for rotated augmentation
        diag = 724
        for i in range(img_for_rotation):
            start_x = random.randint(0, shape[1] - diag)
            start_y = random.randint(0, shape[0] - diag)
            sub_depth = depth[start_y: start_y+diag, start_x: start_x + diag]
            sub_target = target[start_y: start_y+diag, start_x: start_x + diag]
            folder_g = './data/depth/train/'
            folder_m = './data/target/train/'
            plt.imsave(folder_g + 'image{:03d}_big_{:d}.png'.format(idx, i), sub_depth, cmap='viridis')
            plt.imsave(folder_m + 'image{:03d}_big_{:d}.png'.format(idx, i), sub_target, cmap='gray')
            image_counter_rot += 1

#calculate augmentation factors
original_images = image_counter + image_counter_rot
needed_augments = int(original_images / original_data_percentage) - original_images
rot_augments = int(needed_augments * rotational_augment_portion)
normal_augments = needed_augments - rot_augments
rot_aug_factor = int(rot_augments / image_counter_rot)
aug_factor = int(normal_augments / image_counter)
generated_augments = rot_aug_factor * image_counter_rot + (aug_factor) * image_counter
while generated_augments < needed_augments:
    diff = needed_augments - generated_augments
    if (1 - diff / image_counter) < 1 - (diff / image_counter_rot):
        aug_factor += 1
    else:
        rot_aug_factor += 1
    generated_augments = rot_aug_factor * image_counter_rot + (aug_factor) * image_counter
aug_factor = max(1, aug_factor)
rot_aug_factor = max(1, rot_aug_factor)
generated_augments = rot_aug_factor * image_counter_rot + (aug_factor) * image_counter

print('extracted {} 512x512 images and {} 724x724 images'.format(image_counter, image_counter_rot))
print('needed augments: {}'.format(needed_augments))
print('normal augmentation factor: {}'.format(aug_factor))
print('rotational augmentation factor: {}'.format(rot_aug_factor))
print('{} augments are generated, and {}  will be randomly deleted'.format(generated_augments, generated_augments - needed_augments))
print('training data size: {}'.format(original_images + needed_augments))
print('with {} original and {} augmented images'.format(original_images, needed_augments))
print('original data percentage: {}'.format(original_images / (original_images + needed_augments) ))
num_augments = aug_factor
num_augments_rot = rot_aug_factor

quit()

#augment images
print('augmenting images...')
# if num_augments_rot <= 0:
#     num_augments_rot = 1
ia.seed(1234)
sometimes = lambda aug: iaa.Sometimes(0.5, aug)
for idx, img_file in enumerate(tqdm(sorted(glob.glob('./data/depth/train/*.png'), key=numericalSort))):
    img_grain = cv2.imread(img_file)
    img_mask = cv2.imread(img_file.replace('depth', 'target'))
    img_mask = cv2.cvtColor(img_mask, cv2.COLOR_BGR2GRAY)
    img2 = np.zeros((np.array(img_mask).shape[0], np.array(img_mask).shape[1], 1))
    img2[:, :, 0] = img_mask  # same value in each channel
    i1 = []
    i2 = []
    if img_grain.shape[0] > 512:
        for i in range(num_augments_rot):
            i1.append(img_grain)
            i2.append((img2))
    else:
        for i in range(num_augments):
            i1.append(img_grain)
            i2.append((img2))
    i1 = np.asarray(i1)
    i2 = np.asarray(i2)
    i2 = i2.astype(np.int32)
    seq = iaa.Sequential(
        [
            iaa.SomeOf((4, 8), [
                iaa.HorizontalFlip(1.0),
                iaa.VerticalFlip(1.0),
                iaa.AdditiveGaussianNoise(loc=0, scale=(0.0, 0.05*255), per_channel=0.5), # add gaussian noise to images
                iaa.Add((-20, 20), per_channel=0.5), # change brightness of images (by -10 to 10 of original value)
                iaa.Sharpen(alpha=(0, 1.0), lightness=(0.75, 1.5)), # sharpen images
                iaa.OneOf([
                    iaa.GaussianBlur((0, 2.0)), # blur images with a sigma between 0 and 3.0
                    iaa.AverageBlur(k=(2, 7)), # blur image using local means with kernel sizes between 2 and 7
                    iaa.MedianBlur(k=(3, 11)), # blur image using local medians with kernel sizes between 2 and 7
                ]),
                iaa.OneOf([
                    iaa.ElasticTransformation(alpha=20, sigma=2),
                    iaa.ElasticTransformation(alpha=30, sigma=3),
                    iaa.ElasticTransformation(alpha=40, sigma=4),
                    iaa.ElasticTransformation(alpha=50, sigma=5)
                ]),
                iaa.AddToHueAndSaturation((-5, 5), per_channel=True),
                iaa.LogContrast(gain=(0.7, 1.1), per_channel=True),
                sometimes(iaa.PiecewiseAffine(scale=(0.01, 0.05))), # sometimes move parts of the image around
                sometimes(iaa.PerspectiveTransform(scale=(0.01, 0.1))),
                iaa.Crop((0, 50))
                ], random_order=True)  # apply augmenters in random order
        ], random_order=True)
    seq_big = iaa.Sequential([
        iaa.Affine(rotate=(-90, 90)),
        iaa.SomeOf((4, 8), [
                iaa.HorizontalFlip(1.0),
                iaa.VerticalFlip(1.0),
                iaa.AdditiveGaussianNoise(loc=0, scale=(0.0, 0.05*255), per_channel=0.5), # add gaussian noise to images
                iaa.Add((-20, 20), per_channel=0.5), # change brightness of images (by -10 to 10 of original value)
                iaa.Sharpen(alpha=(0, 1.0), lightness=(0.75, 1.5)), # sharpen images
                iaa.OneOf([
                    iaa.GaussianBlur((0, 2.0)), # blur images with a sigma between 0 and 3.0
                    iaa.AverageBlur(k=(2, 7)), # blur image using local means with kernel sizes between 2 and 7
                    iaa.MedianBlur(k=(3, 11)), # blur image using local medians with kernel sizes between 2 and 7
                ]),
                iaa.OneOf([
                    iaa.ElasticTransformation(alpha=20, sigma=2),
                    iaa.ElasticTransformation(alpha=30, sigma=3),
                    iaa.ElasticTransformation(alpha=40, sigma=4),
                    iaa.ElasticTransformation(alpha=50, sigma=5)
                ]),
                iaa.AddToHueAndSaturation((-5, 5), per_channel=True),
                iaa.LogContrast(gain=(0.7, 1.1), per_channel=True),
                sometimes(iaa.PiecewiseAffine(scale=(0.01, 0.05))), # sometimes move parts of the image around
                sometimes(iaa.PerspectiveTransform(scale=(0.01, 0.1))),
                iaa.Crop((0, 50))
                ], random_order=True)  # apply augmenters in random order
    ], random_order=True)

    if img_grain.shape[0] > 512:
        images_aug, segmaps_aug = seq_big(images=i1, segmentation_maps=i2)
    else:
        images_aug, segmaps_aug = seq(images=i1, segmentation_maps=i2)
    folder_g = './data/depth/train/'
    folder_m = './data/target/train/'
    if img_grain.shape[0] <= 512:
        g_out = images_aug[0]
        m_out = segmaps_aug[0]
        if g_out.shape[0] == 512 and g_out.shape[1] == 512 and m_out.shape[0] == 512 and m_out.shape[1] == 512:
            cv2.imwrite(folder_g + "depth_"+str(idx)+"_"+str(0)+'.png', img_grain)
            cv2.imwrite(folder_m + "target_"+str(idx)+"_"+str(0)+'.png', img_mask)
    else:
        g_out = img_grain[107:619, 107:619]
        m_out = img_mask[107:619, 107:619]
        if g_out.shape[0] == 512 and g_out.shape[1] == 512 and m_out.shape[0] == 512 and m_out.shape[1] == 512:
            cv2.imwrite(folder_g+"depth_"+str(idx)+"_"+str(i)+'.png', img_grain[107:619, 107:619])  # write all changed images
            cv2.imwrite(folder_m+"target_"+str(idx)+"_"+str(i)+'.png', img_mask[107:619, 107:619])  # write all changed images

    if img_grain.shape[0] > 512:
        for i in range(num_augments_rot):
            g_out = images_aug[i][107:619, 107:619]
            m_out = segmaps_aug[i][107:619, 107:619]
            if g_out.shape[0] == 512 and g_out.shape[1] == 512 and m_out.shape[0] == 512 and m_out.shape[1] == 512:
                cv2.imwrite(folder_g+"depth_"+str(idx)+"_"+str(i)+'new.png', images_aug[i][107:619, 107:619])  # write all changed images
                cv2.imwrite(folder_m+"target_"+str(idx)+"_"+str(i)+'new.png', segmaps_aug[i][107:619, 107:619])  # write all changed images
    else:
        for i in range(num_augments):
            g_out = images_aug[i]
            m_out = segmaps_aug[i]
            if g_out.shape[0] == 512 and g_out.shape[1] == 512 and m_out.shape[0] == 512 and m_out.shape[1] == 512:
                cv2.imwrite(folder_g + "depth_"+str(idx)+"_"+str(i)+'new.png', images_aug[i])  # write all changed images
                cv2.imwrite(folder_m + "target_"+str(idx)+"_"+str(i)+'new.png', segmaps_aug[i])  # write all changed images
    os.remove(img_file.replace('depth','target'))
    os.remove(img_file)

#random pick files and delete until exact augmentation number is reached
files_to_delete = generated_augments - needed_augments
for i in range(files_to_delete):
    filename = random.choice(glob.glob('./data/depth/train/*new.png'))
    os.remove(filename.replace('depth','target'))
    os.remove(filename)
