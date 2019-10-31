# %% import necessary libraries

import matplotlib.pyplot as plt
import numpy as np
import cv2
import os

from PIL import Image

# %% Create Class


class plateDetection:

    def __init__(self, image):
        self.image = image
        self.img = cv2.cvtColor(self.image, cv2.COLOR_BGR2RGB)
        self.vectorized = self.img.reshape((-1, 3))
        self.vectorized = np.float32(self.vectorized)

    def k_means(self, k):
        criteria = (cv2.TERM_CRITERIA_EPS +
                    cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
        attempts = 10
        _, label, center = cv2.kmeans(self.vectorized, k, None, criteria,
                                      attempts, cv2.KMEANS_PP_CENTERS)
        center = np.uint8(center)
        res = center[label.flatten()]
        result_image = res.reshape((self.image.shape))
        plt.figure(figsize=(10, 6))
        plt.imshow(self.img)
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
            cv2.imwrite('kmeans_results_k=%i/center %i.png' % (k, c), new_image)
        print('Done')


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
            print('waiting to be implemented...')


# %% Create class instance and process sample image

# sample image credit: https://whyy.org/articles/illegal-parkers-beware-philadelphia-wants-you-out-of-the-crosswalk-and-off-the-sidewalk/

philly_car = cv2.imread('phillycar.png')

plate_detect = plateDetection(philly_car)

plate_detect.segmentation()

# %%







