"""
Some codes from https://github.com/Newmu/dcgan_code
"""
from __future__ import division
import math
import json
import random
import pprint
import scipy.misc
import numpy as np
from time import gmtime, strftime

pp = pprint.PrettyPrinter()

get_stddev = lambda x, k_h, k_w: 1/math.sqrt(k_w*k_h*x.get_shape()[-1])

# -----------------------------
# new added functions for pix2pix

def load_data(data, batch_size, fine_size, load_size, flip=True):
    list = []
    for i in range(batch_size):
        img_A, img_B = load_image(data)
        img_A, img_B = preprocess_A_and_B(img_A, img_B, fine_size, load_size, flip=flip, is_test=False)

        img_A = img_A/127.5 - 1.  # color value unify
        img_B = img_B/127.5 - 1.

        img_AB = np.dstack((img_A, img_B))
        # img_AB shape: (fine_size, fine_size, input_c_dim + output_c_dim)
        list.append(img_AB)
    return list

def load_testdata(data, fine_size):
    list = []
    step = int(fine_size / 2)
    ymax = data.shape[0]
    xmax = data.shape[1]
    x = 0
    y = 0
    while x < data.shape[1]:
        while y < data.shape[0]:
            img = data[y:y+step, x:x+step]
            img = scipy.misc.imresize(img, [fine_size, fine_size])
            img = img/127.5 - 1
            img_A = img
            img_A[0:step, 0:step] = -1 
            list.append(np.dstack((img_A, img)))
            img_A = img_A[0:step, step+1:] = -1
            list.append(np.dstack((img_A, img)))
            img_A = img_A[step+1:, 0:step] = -1
            list.append(np.dstack((img_A, img)))
            img_A = img_A[step+1:, step+1:] = -1
            list.append(np.dstack((img_A, img)))
            y += step
        x += step
    return list

def load_image(data):
    xymin = 16
    ymax = data.shape[0]
    xmax = data.shape[1]
    # crop sample from input image
    x = int(np.ceil(np.random.uniform(0, xmax-xymin)))
    y = int(np.ceil(np.random.uniform(0, ymax-xymin)))
    w = int(np.ceil(np.random.uniform(xymin, xmax-x)))
    h = int(np.ceil(np.random.uniform(xymin, ymax-y)))
    img_B = data[y:y+h, x:x+w].copy()
    img_A = img_B.copy()
    # add mask
    minMsk = 4
    ymax = img_B.shape[0]
    xmax = img_B.shape[1]
    mx = int(np.ceil(np.random.uniform(0, xmax-minMsk)))
    my = int(np.ceil(np.random.uniform(0, ymax-minMsk)))
    mw = int(np.ceil(np.random.uniform(minMsk, min(xmax-mx, int(xmax/2)))))
    mh = int(np.ceil(np.random.uniform(minMsk, min(ymax-my, int(ymax/2)))))
    img_A[my:my+mh, mx:mx+mw] = 0

    return img_A, img_B

def preprocess_A_and_B(img_A, img_B, fine_size, load_size, flip=True, is_test=False):
    if is_test:
        img_A = scipy.misc.imresize(img_A, [fine_size, fine_size])
        img_B = scipy.misc.imresize(img_B, [fine_size, fine_size])
    else:
        img_A = scipy.misc.imresize(img_A, [load_size, load_size])
        img_B = scipy.misc.imresize(img_B, [load_size, load_size])

        h1 = int(np.ceil(np.random.uniform(1e-2, load_size-fine_size)))
        w1 = int(np.ceil(np.random.uniform(1e-2, load_size-fine_size)))
        img_A = img_A[h1:h1+fine_size, w1:w1+fine_size]
        img_B = img_B[h1:h1+fine_size, w1:w1+fine_size]

        if flip and np.random.random() > 0.5:
            img_A = np.fliplr(img_A)
            img_B = np.fliplr(img_B)

    return img_A, img_B

# -----------------------------

def get_image(image_path, image_size, is_crop=True, resize_w=64, is_grayscale = False):
    return transform(imread(image_path, is_grayscale), image_size, is_crop, resize_w)

def save_images(images, size, image_path):
    return imsave(inverse_transform(images), size, image_path)

def imread(path, is_grayscale = False):
    if (is_grayscale):
        return scipy.misc.imread(path, flatten = True).astype(np.float)
    else:
        return scipy.misc.imread(path).astype(np.float)

def merge_images(images, size):
    return inverse_transform(images)

def merge(images, size):
    h, w = images.shape[1], images.shape[2]
    img = np.zeros((h * size[0], w * size[1], 3))
    for idx, image in enumerate(images):
        i = idx % size[1]
        j = idx // size[1]
        img[j*h:j*h+h, i*w:i*w+w, :] = image

    return img

def imsave(images, size, path):
    return scipy.misc.imsave(path, merge(images, size))

def transform(image, npx=64, is_crop=True, resize_w=64):
    # npx : # of pixels width/height of image
    if is_crop:
        cropped_image = center_crop(image, npx, resize_w=resize_w)
    else:
        cropped_image = image
    return np.array(cropped_image)/127.5 - 1.

def inverse_transform(images):
    return (images+1.)/2.


