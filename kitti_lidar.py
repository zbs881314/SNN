import pykitti  # install using pip install pykitti
import os
import numpy as np

# Chose which visualization library to use:  "mayavi" or "matplotlib"
#VISLIB = "mayavi"
VISLIB = "matplotlib"

# Raw Data directory information
basedir = '/home/bob/Downloads/data/'
date = '2011_09_26'
drive = '0009'

# Optionally, specify the frame range to load
# since we are only visualizing one frame, we will restrict what we load
# Set to None to use all the data
frame_range = range(150, 151, 1)

# Load the data
dataset = pykitti.raw(basedir, date, drive, frames=frame_range)

# Load Lidar Data
#dataset.get_velo(2)  # Each scan is a Nx4 array of [x,y,z,reflectance]

# Plot only the ith frame (out of what has been loaded)
#i = 0

# def viz_mayavi(points, vals='distance'):
#     x = points[:, 0]
#     y = points[:, 1]
#     z = points[:, 2]
#     d = np.sqrt(x**2 + y**3)
#
#     import mayavi.mlab
#
#     if vals == 'height':
#         col = z
#     else:
#         col = d
#     fig = mayavi.mlab.figure(bgcolor=(0, 0, 0), size=(640, 360))
#     mayavi.mlab.points3d(x, y, z,
#                          col,
#                          model='point',
#                          colormap='spectral',
#     figure=fig)
#     mayavi.mlab.show()
for velo in dataset.velo:

    if VISLIB == "mayavi":
        # Plot using mayavi -Much faster and smoother than matplotlib
        import mayavi.mlab


        fig = mayavi.mlab.figure(bgcolor=(0, 0, 0), size=(640, 360))
        mayavi.mlab.points3d(
            velo[:, 0],  # x
            velo[:, 1],  # y
            velo[:, 2],  # z
            velo[:, 2],  # Height data used for shading
            mode="point",  # How to render each point {'point', 'sphere' , 'cube' }
            colormap='spectral',  # 'bone', 'copper',
            # color=(0, 1, 0),     # Used a fixed (r,g,b) color instead of colormap
            scale_factor=100,  # scale of the points
            line_width=10,  # Scale of the line, if any
            figure=fig,
        )
        # velo[:, 3], # reflectance values
        mayavi.mlab.show()

    else:
        # Plot Using Matplotlib - Much slower than mayavi.
        # NOTE: Only 1 out of every 100 points are plotted using the matplotlib
        #       version to prevent crashing the computer
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d import Axes3D

        skip = 10  # plot one in every `skip` points

        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        print(velo.shape[0])
        velo_range = range(0, velo.shape[0], skip)  # skip points to prevent crash
        print(velo[velo_range, 0])
        ax.scatter(velo[velo_range, 0],  # x
                   velo[velo_range, 1],  # y
                   velo[velo_range, 2],  # z
                   c=velo[velo_range, 3],  # reflectance
                   cmap='gray',
                   )
        ax.set_title('Lidar scan (subsampled)')
        ax.axis('scaled')
        plt.show()