# %% import necessary libraries

import matplotlib.pyplot as plt
import numpy as np
import cv2
import os
from ast import literal_eval as make_tuple

from PIL import Image

# %% Create Class

'''
    References:
    https://towardsdatascience.com/introduction-to-image-segmentation-with-k-means-clustering-83fd0a9e2fc3
    Philly car image: https://whyy.org/articles/illegal-parkers-beware-philadelphia-wants-you-out-of-the-crosswalk-and-off-the-sidewalk/
    graph cut: size limitation: https://julie-jiang.github.io/image-segmentation/
    https://www.amsterdam.nl/en/parking/on-street-parking/
    https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_imgproc/py_grabcut/py_grabcut.html
    https://github.com/opencv/opencv/blob/master/samples/python/grabcut.py
'''


class plateDetection:

    def __init__(self, image):
        self.image = image
        self.image = cv2.cvtColor(self.image, cv2.COLOR_BGR2RGB)

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
        plt.figure(figsize=(10, 6))
        plt.imshow(img)
        plt.title('Original Image')
        plt.xticks([]), plt.yticks([])

        plt.figure(figsize=(10, 6))
        plt.imshow(result_image)
        plt.title('Segmented Image when K = %i' % k)
        plt.xticks([]), plt.yticks([])

        # trying to create and save new image, but k-means'result isn't
        # quite suitable for this
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
            cv2.imwrite('kmeans_results_k=%i/center_%i.png' % (k, c), new_image)
        print('Done')

    def graph_cut(self):
        try:
            os.mkdir('graphCut_results/')
        except FileExistsError:
            pass
        mode = input('use rect or mask? ')
        img = self.image
        mask = np.zeros(img.shape[:2], np.uint8)
        bgdModel = np.zeros((1, 65), np.float64)
        fgdModel = np.zeros((1, 65), np.float64)
        if mode == 'rect':
            # rect = (520, 350, 830, 1200)
#             rect = (300, 350, 520, 600)
            rect = make_tuple(input(
                    'Enter a rect coordinate, use tuple (a,b,c,d): '))
            cv2.grabCut(img, mask, rect, bgdModel, fgdModel,
                        5, cv2.GC_INIT_WITH_RECT)
            mask2 = np.where((mask == 2) | (mask == 0), 0, 1).astype('uint8')
            img = img * mask2[:, :, np.newaxis]
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            plt.figure(figsize=(16, 8))
            plt.imshow(img), plt.show()
            cv2.imwrite('graphCut_results/rect_seg.jpg', img)

        elif mode == 'mask':
            path = input('Indicate mask path...')
            newmask = cv2.imread(path, 0)
            # whereever it is marked white (sure foreground), change mask=1
            # whereever it is marked black (sure background), change mask=0
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
            cv2.imwrite('graphCut_results/mask_seg.jpg', img)
            # if you mask from last 'if' we can add mask results to rect result
        else:
            print('Please try again...')

    def segmentation(self):
        method = 'n/a'
        method_list = ['kmeans', 'mean shift', 'normalized cut', 'graph cut']
        while method not in method_list:
            method = input('please indicate a segmentation method: choose ' +
                           '"kmeans", "mean shift", "normalized cut", or' +
                           '"graph cut": \n')

        if method == 'kmeans':
            k = input('choose your k: ')
            self.k_means(int(k))

        elif method == 'mean shift':
            print('Will implement if I had time...')

        elif method == 'normalized cut':
            print('waiting to be implemented...')

        elif method == 'graph cut':
            self.graph_cut()

# %% Create class instance and process sample image
#
teaser = cv2.imread('teaser.png')

plate_detect = plateDetection(teaser)

plate_detect.segmentation()
## %%
#ams = cv2.imread('amsterdam.jpg')
#
#plate_detect = plateDetection(ams)
#
#plate_detect.segmentation()
#



