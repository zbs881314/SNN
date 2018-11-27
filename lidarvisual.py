import pykitti  # install using pip install pykitti
import os
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Raw Data directory information
basedir = '/home/bob/Downloads/data/'
date = '2011_09_26'
drive = '0009'

def convert_xyz_to_2d(places):
    theta = np.arctan2(places[:, 1], places[:, 0])
    ave_theta = np.average(theta)
    phi = np.arctan2(places[:, 2], np.sqrt(places[:, 0]**2 + places[:, 1]**2))
    ave_phi = np.average(phi)
    r = (theta / ave_theta).astype(np.int32)
    c = (phi / ave_phi).astype(np.int32)
    d = np.sqrt(places[:, 0]**2 + places[:, 1]**2)
    print("places", places.shape)
    print(np.max(places, axis=0))
    print(np.min(places, axis=0))
    print("theta", theta.shape)
    print(theta.max(axis=0))
    print(theta.min(axis=0))
    print(ave_theta)
    print("phi", phi.shape)
    print(phi.min())
    print(phi.max())
    print(ave_phi)
    print(r.max(), r.min(), c.max(), c.min())
    # fig = plt.figure()
    # # ax = fig.add_subplot(1, 1, 1)
    # plt.scatter(r, c)
    # plt.ion()
    # plt.show()
    plt.hist(phi)
    plt.show()
    plt.scatter(d, places[:, 2])
    plt.show()



frame_range = range(150, 151, 1)
print(frame_range)

# Load the data
dataset = pykitti.raw(basedir, date, drive, frames=frame_range)

for places in dataset.velo:
    convert_xyz_to_2d(places)

for velo in dataset.velo:

    skip = 10  # plot one in every `skip` points

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    #print(velo.shape[0])
    velo_range = range(0, velo.shape[0], skip)  # skip points to prevent crash
    #print(velo[velo_range, 0])
    ax.scatter(velo[velo_range, 0],  # x
               velo[velo_range, 1],   # y
               velo[velo_range, 2],   #  z
               c=velo[velo_range, 3],  # reflectance
               cmap='gray')
    ax.set_title('Lidar scan (subsampled)')
    plt.show()