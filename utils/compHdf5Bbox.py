import os
import h5py
import argparse
import numpy as np
from matplotlib import pyplot as plt
import sys
import json

import math
from PIL import Image
from numpy.ma.core import get_data
import csv
# ----------------------------------------------------------------------------------------------------------------

bounding_box_dataset = np.loadtxt("C:\\Users\\Juan\\Desktop\\CiCLAB\\Juan_Segmentacion\\YOLO_Dataset\\000000.txt", usecols = range(5))
id_group = np.array(bounding_box_dataset[:, 0])
x = np.array(bounding_box_dataset[:, 1])
y = np.array(bounding_box_dataset[:, 2])
width = np.array(bounding_box_dataset[:, 3])
height = np.array(bounding_box_dataset[:, 4])

id = []
with open('C:\\Users\\Juan\\Desktop\\CiCLAB\\Juan_Segmentacion\\Blender_Dataset\\0000\\class_inst_col_map.csv','r') as csv_file:
    i = 0
    csv_read = csv.reader(csv_file)
    for row in csv_read:
        if i >= 2:
            id.insert(i - 2, int(row[0]))
        i = i + 1
id = np.array(id)
id = np.ma.getdata((np.ma.round(np.ma.array(id))).astype(int))

corner_up_lft = np.zeros((id_group.size, 2))
corner_up_rgt = np.zeros((id_group.size, 2))
corner_dw_lft = np.zeros((id_group.size, 2))
corner_dw_rgt = np.zeros((id_group.size, 2))

for i in range(id_group.size):
    corner_up_lft[i][0] = (x.item(i) - width.item(i)/2) * 1920/1080
    corner_up_lft[i][1] = y.item(i) + height.item(i)/2

    corner_up_rgt[i][0] = (x.item(i) + width.item(i)/2) * 1920/1080
    corner_up_rgt[i][1] = y.item(i) + height.item(i)/2

    corner_dw_lft[i][0] = (x.item(i) - width.item(i)/2) * 1920/1080
    corner_dw_lft[i][1] = y.item(i) - height.item(i)/2

    corner_dw_rgt[i][0] = (x.item(i) + width.item(i)/2) * 1920/1080
    corner_dw_rgt[i][1] = y.item(i) - height.item(i)/2

corner_up_lft = np.multiply(corner_up_lft, 1080)
corner_up_rgt = np.multiply(corner_up_rgt, 1080)
corner_dw_lft = np.multiply(corner_dw_lft, 1080)
corner_dw_rgt = np.multiply(corner_dw_rgt, 1080)

corner_up_lft_int = np.ma.getdata((np.ma.round(np.ma.array(corner_up_lft))).astype(int))
corner_up_rgt_int = np.ma.getdata((np.ma.round(np.ma.array(corner_up_rgt))).astype(int))
corner_dw_lft_int = np.ma.getdata((np.ma.round(np.ma.array(corner_dw_lft))).astype(int))
corner_dw_rgt_int = np.ma.getdata((np.ma.round(np.ma.array(corner_dw_rgt))).astype(int))

beta = np.zeros((id_group.size, id_group.size))
for i in range(id_group.size):
    for j in range(id_group.size):
        if (i != j):
            if ( ((corner_up_lft[i][0] < corner_up_lft[j][0] < corner_dw_rgt[i][0]) and 
            (corner_up_lft[i][1] > corner_up_lft[j][1] > corner_dw_rgt[i][1])) or 

            ((corner_up_lft[i][0] < corner_up_rgt[j][0] < corner_dw_rgt[i][0]) and 
            (corner_up_lft[i][1] > corner_up_rgt[j][1] > corner_dw_rgt[i][1])) or 

            ((corner_up_lft[i][0] < corner_dw_lft[j][0] < corner_dw_rgt[i][0]) and 
            (corner_up_lft[i][1] > corner_dw_lft[j][1] > corner_dw_rgt[i][1])) or 

            ((corner_up_lft[i][0] < corner_dw_rgt[j][0] < corner_dw_rgt[i][0]) and 
            (corner_up_lft[i][1] > corner_dw_rgt[j][1] > corner_dw_rgt[i][1])) ):
                beta[i][j] = 1

bb_int = beta.nonzero()
bb_unique_int = []
for i in [0,1]:
    bb_unique_int = np.append(bb_unique_int, np.unique(bb_int[i]))
    bb_unique_int = np.unique(bb_unique_int)
bb_unique_int = np.ma.getdata((np.ma.round(np.ma.array(bb_unique_int))).astype(int))

data = h5py.File('C:\\Users\\Juan\\Desktop\\CiCLAB\\Juan_Segmentacion\\Blender_Dataset\\0000\\0.hdf5','r')
segmap = data['segmap']
segmap_img = segmap[:, :, 0]
segmap_img = np.ma.getdata((np.ma.round(np.ma.array(segmap_img))).astype(int))

bb_segmap_cnt = np.zeros([bb_unique_int.size, 13])
bb_segmap_cnt = np.ma.getdata((np.ma.round(np.ma.array(bb_segmap_cnt))).astype(int))

for i in range(bb_unique_int.size):
    for h in range(corner_dw_rgt_int[bb_unique_int[i]][1] - 1, corner_up_lft_int[bb_unique_int[i]][1] - 1, 1):
        for w in range(corner_up_lft_int[bb_unique_int[i]][0] - 1, corner_dw_rgt_int[bb_unique_int[i]][0] - 1, 1):
            bb_segmap_cnt[i][segmap_img[h][w] - 1] = bb_segmap_cnt[i][segmap_img[h][w] - 1] + 1
bb_segmap_cnt_wo_bckgnd = (bb_segmap_cnt[:, 1:bb_segmap_cnt.size]).astype(float)
for i in range(bb_segmap_cnt_wo_bckgnd.shape[0]):
    bb_segmap_cnt_wo_bckgnd[i] = np.divide(bb_segmap_cnt_wo_bckgnd[i], np.sum(bb_segmap_cnt_wo_bckgnd[i]))

exclude= []
for i in range(bb_unique_int.size):
    if bb_segmap_cnt_wo_bckgnd[i][id[bb_unique_int[i]] - 2] < 0.65:
        exclude = np.append(exclude, id[bb_unique_int[i]])
exclude = np.ma.getdata((np.ma.round(np.ma.array(exclude))).astype(int))
print(exclude)

# plt.axes()
    # plt.imshow(segmap_img)
    # for i in range(id_group.size):
    #     rectangle = plt.Rectangle((x.item(i)*1920 - width.item(i)*960, y.item(i)*1080 - height.item(i)*540), width.item(i)*1920, height.item(i)*1080, fc=(0, 0, 1, 0.5), ec="red")
    #     plt.gca().add_patch(rectangle)
    # plt.scatter(np.multiply(x,1920), np.multiply(y,1080), fc='white', ec="red")
    # plt.axis('scaled')
    # plt.show()