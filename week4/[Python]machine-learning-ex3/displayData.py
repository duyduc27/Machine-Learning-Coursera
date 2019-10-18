# import random
# import numpy as np
# import matplotlib.pyplot as plt

# #from show import show
# def getDatumImg(row):

#     """
#     Function that is handed a single np array with shape 1x400,
#     crates an image object from it, and returns it
#     """
#     width, height = 20, 20
#     square = row[1:].reshape(width, height)
#     return square.T

# def displayData(x):
#     """
#     Function that picks 100 random rows from X, creates a 20x20 image from each,
#     then stitches them together into a 10x10 grid of images, and shows it.
#     """
#     width, height = 20, 20
#     nrows, ncols = 10, 10

#     # if is used to visualize hidden layer
#     if x.shape[0] < nrows * ncols:
#         nrows, ncols = 5, 5

#     indices_to_display = random.sample(range(0, x.shape[0]), nrows * ncols)

#     big_picture = np.zeros((height * nrows, width * ncols))

#     irow, icol = 0, 0

#     for idx in indices_to_display:
#         if icol == ncols:
#             irow += 1
#             icol = 0

#         iimg = getDatumImg(x[idx])
#         big_picture[irow * height:irow * height + iimg.shape[0], icol * width:icol * width + iimg.shape[1]] = iimg

#         icol += 1

#     plt.imshow(big_picture, cmap=cm.Greys_r)
#     plt.axis('off')
#     plt.show()