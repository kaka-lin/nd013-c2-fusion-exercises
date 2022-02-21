# ---------------------------------------------------------------------
# Exercises from lesson 2 (object detection)
# Copyright (C) 2020, Dr. Antje Muntzinger / Dr. Andreas Haja.
#
# Purpose of this file : Starter Code
#
# You should have received a copy of the Udacity license together with this program.
#
# https://www.udacity.com/course/self-driving-car-engineer-nanodegree--nd013
# ----------------------------------------------------------------------
#

from PIL import Image
import io
import sys
import os
import cv2
import open3d as o3d
import math
import numpy as np
import zlib

import matplotlib
matplotlib.use('agg') # change backend so that figure maximizing works on Mac as well
import matplotlib.pyplot as plt

# Exercise C2-4-6 : Plotting the precision-recall curve
def plot_precision_recall():

    # Please note: this function assumes that you have pre-computed the precions/recall value pairs from the test sequence
    #              by subsequently setting the variable configs.conf_thresh to the values 0.1 ... 0.9 and noted down the results.

    # Please create a 2d scatter plot of all precision/recall pairs
    pass



# Exercise C2-3-4 : Compute precision and recall
def compute_precision_recall(det_performance_all, conf_thresh=0.5):

    if len(det_performance_all)==0 :
        print("no detections for conf_thresh = " + str(conf_thresh))
        return

    # extract the total number of positives, true positives, false negatives and false positives
    # format of det_performance_all is [ious, center_devs, pos_negs]

    #print("TP = " + str(true_positives) + ", FP = " + str(false_positives) + ", FN = " + str(false_negatives))

    # compute precision

    # compute recall

    #print("precision = " + str(precision) + ", recall = " + str(recall) + ", conf_thres = " + str(conf_thresh) + "\n")




# Exercise C2-3-2 : Transform metric point coordinates to BEV space
def pcl_to_bev(lidar_pcl, configs, vis=True):

    ####### Discretize Feature Map
    # compute bev-map discretization by dividing x-range by the bev-image height
    # positive-x pointing up and configs.bev_height is the number of pixels in height direction
    # bev_discret is in meters per pixel
    bev_discret = (configs.lim_x[1] - configs.lim_x[0]) / configs.bev_height

    # create a copy of the lidar pcl and transform all metrix x-coordinates into bev-image coordinates
    lidar_pcl_cpy = np.copy(lidar_pcl)
    lidar_pcl_cpy[:, 0] = np.int_(np.floor(lidar_pcl_cpy[:, 0] / bev_discret))

    # transform all metrix y-coordinates as well but center the foward-facing x-axis on the middle of the image
    # y ranges between [-range, range]
    lidar_pcl_cpy[:, 1] = np.int_(np.floor(lidar_pcl_cpy[:, 1] / bev_discret) + (configs.bev_width + 1) / 2)

    # shift level of ground plane to avoid flipping from 0 to 255 for neighboring pixels
    lidar_pcl_cpy[:, 2] = lidar_pcl_cpy[:, 2] - configs.lim_z[0]

    ####### Create height map
    # re-arrange elements in lidar_pcl_cpy by sorting first by x, then y, then by decreasing height
    idx_height = np.lexsort((-lidar_pcl_cpy[:, 2], lidar_pcl_cpy[:, 1], lidar_pcl_cpy[:, 0]))
    lidar_pcl_hei = lidar_pcl_cpy[idx_height]

    # extract all points with identical x and y such that only the top-most z-coordinate is kept (use numpy.unique)
    # height Hij = max(Pij * [0,0,1]^T) -> 只保留最高的點(z軸)
    _, idx_height_unique = np.unique(lidar_pcl_hei[:, 0:2], axis=0, return_index=True)
    lidar_pcl_hei = lidar_pcl_hei[idx_height_unique]

    # create the height map
    # assign the height value of each unique entry in "lidar_pcl_hei" to the height map and
    # make sure that each entry is normalized on the difference between the upper and lower height defined in the config file
    # pcl: (x, y, z) -> 2d image: (h, w)
    # x in the data is front -> image coordinates y (h)
    # y in the data is left -> image coordinates x (w)
    height_map = np.zeros((configs.bev_height + 1, configs.bev_width + 1))
    height_map[np.int_(lidar_pcl_hei[:, 0]), np.int_(lidar_pcl_hei[:, 1])] = lidar_pcl_hei[:, 2] / float(np.abs(configs.lim_z[1] - configs.lim_z[0]))

    # visualize height map
    # if vis:
    #    img_height = height_map * 256
    #    img_height = img_height.astype(np.uint8)
    #    while (1):
    #        cv2.imshow('img_height', img_height)
    #        if cv2.waitKey(10) & 0xFF == 27:
    #            break
    #    cv2.destroyAllWindows()

    ####### Create intensity map
    # sort points such that in case of identical BEV grid coordinates,
    # the points in each grid cell are arranged based on their intensity
    # Intensity Iij = max(I(Pij) -> 最大值為 1.0
    #
    # 觀察直方圖, code如下:
    #   b = np.array([0, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1, 1e+1, 1e+2, 1e+3, 1e+4, 1e+5, 1e+6, 1e+7])
    #   hist, bins = np.histogram(lidar_pcl[:,3], bins=b)
    #   print(hist)
    # 可以看出大部分 intensity 值落在 0.001 ~ 1.0
    # 且高 intensity 佔不到總量的 1%
    # 因此我們可以安全第將大於 1.0 的值都限制在 1.0
    lidar_pcl_cpy[lidar_pcl_cpy[:, 3] > 1.0, 3] = 1.0
    idx_intensity = np.lexsort((-lidar_pcl_cpy[:, 3], lidar_pcl_cpy[:, 1], lidar_pcl_cpy[:, 0]))
    lidar_pcl_cpy = lidar_pcl_cpy[idx_intensity]

    # only keep one point per grid cell
    _, indices = np.unique(lidar_pcl_cpy[:, 0:2], axis=0, return_index=True)
    lidar_pcl_int = lidar_pcl_cpy[indices]

    # create the intensity map
    intensity_map = np.zeros((configs.bev_height + 1, configs.bev_width + 1))
    intensity_map[np.int_(lidar_pcl_int[:, 0]), np.int_(lidar_pcl_int[:, 1])] = lidar_pcl_int[:, 3] / (np.amax(lidar_pcl_int[:, 3])-np.amin(lidar_pcl_int[:, 3]))

    # visualize intensity map
    if vis:
       img_intensity = intensity_map * 256
       img_intensity = img_intensity.astype(np.uint8)
       while (1):
           cv2.imshow('img_intensity', img_intensity)
           if cv2.waitKey(10) & 0xFF == 27:
               break
       cv2.destroyAllWindows()
