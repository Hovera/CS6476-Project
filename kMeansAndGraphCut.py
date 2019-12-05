# %% import necessary libraries

import matplotlib.pyplot as plt
import numpy as np
import cv2
import os
from ast import literal_eval as make_tuple
import glob

# %% Create Class


class imageSegmentation():

    def __init__(self, image, filename):
        self.filename = filename
        self.image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # this method is very redundant, could've just call the above directly
    # but if you want it to act like a product, this is the choice
    def initalization(self):
        method = 'n/a'
        method_list = ['kmeans', 'graph cut']
        while method not in method_list:
            method = input('please indicate a segmentation method: choose ' +
                           '"kmeans" or "graph cut": \n')

        if method == 'kmeans':
            k = input('choose your k: ')
            self.k_means(int(k))

        elif method == 'graph cut':
            self.graph_cut()

    def k_means(self, k):
        img = self.image
        vectorized = img.reshape((-1, 3))
        vectorized = np.float32(vectorized)
        criteria = (cv2.TERM_CRITERIA_EPS +
                    cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
        attempts = 10
        _, label, center = cv2.kmeans(vectorized, k, None, criteria,
                                      attempts, cv2.KMEANS_PP_CENTERS)
        center = np.uint8(center)
        res = center[label.flatten()]
        result_image = res.reshape((self.image.shape))
#        plt.figure(figsize=(10, 6))
#        plt.imshow(img)
#        plt.title('Original Image')
#        plt.xticks([]), plt.yticks([])
#
#        plt.figure(figsize=(10, 6))
#        plt.imshow(result_image)
#        plt.title('Segmented Image when K = %i' % k)
#        plt.xticks([]), plt.yticks([])

        # save clustered image
        try:
            os.mkdir('kmeans_results_k=%i/' % k)
        except FileExistsError:
            pass
        row, col, _ = result_image.shape
        new_image = np.zeros((row, col, 3), np.uint8)
        for c in range(center.shape[0]):
            new_image = np.zeros((row, col, 3), np.uint8)
            for i in range(0, row):
                for j in range(0, col):
                    if (result_image[i][j] == center[c]).all():
                        new_image[i][j] = center[c]
            cv2.imwrite('kmeans_results_k=%i/%s_center_%i.png' % (
                    k, self.filename, c), new_image)
        print('%s ...Done' % self.filename)

    def graph_cut(self):
        try:
            os.mkdir('graph_cut_results/')
        except FileExistsError:
            pass
        mode = input('use rect or mask? ')
        img = self.image
        mask = np.zeros(img.shape[:2], np.uint8)
        bgdModel = np.zeros((1, 65), np.float64)
        fgdModel = np.zeros((1, 65), np.float64)
        if mode == 'rect':
            # rect = (520, 350, 830, 1200)
            # rect = (300, 350, 520, 600)
            rect = make_tuple(input(
                    'Enter a rect coordinate, use tuple (a,b,c,d): '))
            cv2.grabCut(img, mask, rect, bgdModel, fgdModel,
                        5, cv2.GC_INIT_WITH_RECT)
            mask2 = np.where((mask == 2) | (mask == 0), 0, 1).astype('uint8')
            img = img * mask2[:, :, np.newaxis]
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            plt.figure(figsize=(16, 8))
            plt.imshow(img), plt.show()
            cv2.imwrite('graph_cut_results/%s_rect_seg.jpg' % self.filename,
                        img)

        elif mode == 'mask':
            path = input('Indicate mask path...')
            newmask = cv2.imread(path, 0)
            # wherever it is marked white (sure foreground), change mask=1
            # wherever it is marked black (sure background), change mask=0
            mask[newmask == 0] = 0
            mask[newmask == 255] = 1
            mask, bgdModel, fgdModel = cv2.grabCut(
                    img, mask, None, bgdModel, fgdModel,
                    5, cv2.GC_INIT_WITH_MASK)
            mask3 = np.where((mask == 2) | (mask == 0), 0, 1).astype('uint8')
            img = img * mask3[:, :, np.newaxis]
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            plt.figure(figsize=(16, 8))
            plt.imshow(img), plt.show()
            cv2.imwrite('graph_cut_results/%s_mask_seg.jpg' % self.filename,
                        img)

        else:
            print('Please try again...')


# %% Create class instance and process sample image
teaser = cv2.imread('sample_image/phillycar.png')

plate_detect = imageSegmentation(teaser, 'phillycar')

plate_detect.initialization()

# Another example of sample image

ams = cv2.imread('sample_image/amsterdam.jpg')

plate_detect = imageSegmentation(ams, 'amsterdam')

plate_detect.initialization()


# %% Use kmeans on all images in the dataset

image_list = []
file_name = []

for file in glob.glob('dataset/*.jpg'):
    im = cv2.imread(file)
    image_list.append(im)
    file = file.replace('dataset/', '')
    file = file.replace('.jpg', '')
    file_name.append(file)

# select first 6 images
# image_list = image_list[1:6]
# file_name = file_name[1:6]

for i in range(0, len(image_list)):
    plate_detect = imageSegmentation(image_list[i], file_name[i])
    plate_detect.k_means(k=3)
print('All Done')
