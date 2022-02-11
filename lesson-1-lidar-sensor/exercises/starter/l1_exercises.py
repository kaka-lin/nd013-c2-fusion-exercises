# ---------------------------------------------------------------------
# Exercises from lesson 1 (lidar)
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
import numpy as np
import zlib

## Add current working directory to path
sys.path.append(os.getcwd())

## Waymo open dataset reader
from tools.waymo_reader.simple_waymo_open_dataset_reader import dataset_pb2


# Exercise C1-5-5 : Visualize intensity channel
def vis_intensity_channel(frame, lidar_name):

    print("Exercise C1-5-5")

    # extract range image from frame
    lidar = [obj for obj in frame.lasers if obj.name == lidar_name][0] # get lasers data structure from frame
    ri = [] # ri: ramge_image
    if len(lidar.ri_return1.range_image_compressed) > 0: # use first response
        ri = dataset_pb2.MatrixFloat()
        ri.ParseFromString(zlib.decompress(lidar.ri_return1.range_image_compressed))
        ri = np.array(ri.data).reshape(ri.shape.dims)
    ri[ri<0]=0.0

    # map value intensity to 8bit
    ri_intensity = ri[:,:,1]
    #ri_intensity = ri_intensity * 255 / (np.amax(ri_intensity) - np.amin(ri_intensity))
    # contrast adjustment
    ri_intensity = (np.amax(ri_intensity) / 2) * (ri_intensity * 255 / (np.amax(ri_intensity) - np.amin(ri_intensity)))
    img_intensity = ri_intensity.astype(np.uint8)

    # focus on +/- 45° around the image center
    deg45 = int(img_intensity.shape[1] / 8) # 360° / 45° = 8
    ri_center = int(img_intensity.shape[1]/2)
    img_intensity = img_intensity[:,ri_center-deg45:ri_center+deg45]

    cv2.imshow('intensity image', img_intensity)
    cv2.waitKey(0)


# Exercise C1-5-2 : Compute pitch angle resolution
def print_pitch_resolution(frame, lidar_name):

    print("Exercise C1-5-2")

    # 1. Load range image
    lidar = [obj for obj in frame.lasers if obj.name == lidar_name][0] # get lasers data structure from frame
    ri = [] # ri: ramge_image
    if len(lidar.ri_return1.range_image_compressed) > 0: # use first response
        ri = dataset_pb2.MatrixFloat()
        ri.ParseFromString(zlib.decompress(lidar.ri_return1.range_image_compressed))
        ri = np.array(ri.data).reshape(ri.shape.dims)

    # 2. Compute vertical field-of-view (VFOV) from lidar calibration in rad
    lidar_calib = [obj for obj in frame.context.laser_calibrations if obj.name == lidar_name][0] # get laser calibration
    min_pitch = lidar_calib.beam_inclination_min
    max_pitch = lidar_calib.beam_inclination_max
    vfov_rad = max_pitch - min_pitch

    # compute pitch resolution and convert it to angular minutes
    ## ri: row is pitcn
    pitch_resolution_rad = vfov_rad / ri.shape[0]
    pitch_resolution_deg = pitch_resolution_rad * 180 / np.pi
    print("pitch angle resolution = " + '{0:.2f}'.format(pitch_resolution_deg) + "°")


# Exercise C1-3-1 : print no. of vehicles
def print_no_of_vehicles(frame):

    print("Exercise C1-3-1")

    # find out the number of labeled vehicles in the given frame
    # Hint: inspect the data structure frame.laser_labels
    num_vehicles = 0
    for label in frame.laser_labels:
        if label.type == label.TYPE_VEHICLE:
            num_vehicles += 1

    print("number of labeled vehicles in current frame = " + str(num_vehicles))


# Exercise C1-3-4 : print no. of laser LEDs of the top LiDAR
def print_no_of_laser_leds(frame, lidar_name):
    for laser in frame.context.laser_calibrations:
        if laser.name == lidar_name:
            print("number of individual laser LEDs in current frame = ",
                  len(laser.beam_inclinations))
