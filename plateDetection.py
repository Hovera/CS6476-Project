# %% import necessary libraries

import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import cv2

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
        _, label, center = cv2.kmeans(
                self.vectorized, k, None, criteria, attempts, cv2.KMEANS_RANDOM_CENTERS)
        center = np.uint8(center)
        res = center[label.flatten()]
        result_image = res.reshape((self.image.shape))
        plt.figure(figsize=(16, 8))
        plt.imshow(self.img)
        plt.title('Original Image')
        plt.xticks([]), plt.yticks([])

        plt.figure(figsize=(16, 8))
        plt.imshow(result_image)
        plt.title('Segmented Image when K = %i' % k)
        plt.xticks([]), plt.yticks([])
        # trying to create and save new image, but k-means'result isn't
        # quite suitable for this
#        row, col, _ = result_image.shape
#        new_image = np.zeros((row, col, 3), np.uint8)
#        for i in range(0, row):
#            for j in range(0, col):
#                if (result_image[i][j] == center[4]).all():
#                    new_image[i][j] = center[4]
#        plt.imshow(new_image)

    def segmentation(self):
        method = 'n/a'
        while method not in ['kmeans', 'mean shift', 'normalized cut']:
            method = input('please indicate a segmentation method: choose ' +
                           '"kmeans", "mean shift", or "normalized cut": \n')

        if method == 'kmeans':
            k = input('choose your cluster number: ')
            self.k_means(int(k))

        elif method == 'mean shift':
            print('It\'s like kmeans without initialization so ...')
        elif method == 'normalized cut':
            print('waiting to be implemented...')


# %% Create class instance and process sample image

# sample image credit: https://whyy.org/articles/illegal-parkers-beware-philadelphia-wants-you-out-of-the-crosswalk-and-off-the-sidewalk/

philly_car = cv2.imread('phillycar.png')

plate_detect = plateDetection(philly_car)

plate_detect.segmentation()

# %%







