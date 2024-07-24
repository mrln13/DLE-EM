import sys
import math
import time
import random
import czifile
import tifffile
from utils import *
from tqdm import tqdm
from pathlib import Path
from match_template import *
import matplotlib.pyplot as plt
from skimage.transform import downscale_local_mean


def unique_tuples(x_max, y_max, num_tuples):
    """
    Generates (pseudo)random tuples containing tile coordinates
    :param x_max: int
        maximum X coordinate (px)
    :param y_max: int
        maximum Y coordinate (px)
    :param num_tuples: int
        Number of tuples to create
    :return: set
        set of tuples with X, Y coordinates
    """
    tuples = set()
    while len(tuples) < num_tuples:
        x = random.randint(0, x_max)
        y = random.randint(0, y_max)
        tuples.add((x, y))
    return tuples


def match(im1, im2, factor, plot):
    """
    Extracts, and returns, coordinates of template (im2) within a larger image (im1).
    :param im1: array-like
        the larger image in which to search for the template
    :param im2: array-like
        the template to find within 'im1'
    :param factor: int
        resolution difference factor
    :param plot: boolean
        boolean indicating whether to plot the results
    :return: tuple (containing a tuple and a float)
        coordinates(x1, x2, y1, y2) of the location of the template within im1 and the duration of processing time
    """
    #  Rescale in case of different magnification
    if factor != 1:
        if factor > 1:
            im1 = downscale_local_mean(im1, (factor, factor))
        if factor < 1:
            print(factor)
            factor = 1 / factor
            print(factor)
            im2 = downscale_local_mean(im2, (factor, factor))
    start = time.time()
    result = match_template(im1, im2)
    end = time.time()

    ij = np.unravel_index(np.argmax(result), result.shape)
    x, y = ij[::-1]
    htile, wtile = im2.shape

    # Plot results (use for debugging)
    if plot:
        fig = plt.figure(figsize=(8, 3))
        ax1 = plt.subplot(1, 3, 1)
        ax2 = plt.subplot(1, 3, 2)
        ax3 = plt.subplot(1, 3, 3)  # , sharex=ax2, sharey=ax2)

        ax1.imshow(im2, cmap=plt.cm.gray)
        ax1.set_axis_off()
        ax1.set_title('LR tile')

        ax2.imshow(im1, cmap=plt.cm.gray)
        ax2.set_axis_off()
        ax2.set_title('HR tile')
        # highlight matched region
        rect = plt.Rectangle((x, y), wtile, htile, edgecolor='r', facecolor='none')
        ax2.add_patch(rect)

        ax3.imshow(result)
        ax3.set_axis_off()
        ax3.set_title('Maximum')
        # highlight matched region
        ax3.autoscale(False)
        ax3.plot(x, y, 'o', markeredgecolor='r', markerfacecolor='none', markersize=10)

        plt.show()  # (block=False)

    coordinates = y, y + htile, x, x + wtile
    return coordinates, end - start


def create_pairs(hr_p, lr_p, name, out_path, rgb=False, test_frac=0, padding_frac=0.1):
    """
    Create HR/LR tile pairs from HR and LR maps. If padding_frac > 0, HR images are cropped with extra margin to allow
    for template matching, which significantly improves image pair alignment.
    :param hr_p: string
        path to the HR file
    :param lr_p: string
        path to the LR file
    :param name: string
        base name for the output files
    :param out_path: string
        output directory
    :param rgb: boolean
        boolean indicating whether to process RGB images
    :param test_frac: float (between 0 and 1)
        faction of the dataset to be used as test set
    :param padding_frac: float (between 0 and 1)
        fraction of padding added to HR images for better alignment
    :return: tuple
        number of training image pairs and testing image pairs
    """
    if hr_p.lower().endswith(('.tif', '.tiff')):
        hr = tifffile.imread(hr_p)
        lr = tifffile.imread(lr_p)
        if len(hr.shape) > 2:
            rgb = yes_no_inp('Found multiple channels. Process as RGB data? [Y/N]\n')
        if not rgb and len(hr) == 3:
            hr = hr[:, :, 0]
            lr = lr[:, :, 0]
    elif hr_p.lower().endswith('.czi'):
        hr = czifile.imread(hr_p)
        lr = czifile.imread(lr_p)
        # Extract relevant channel
        if hr.shape[4] > 1:
            rgb = yes_no_inp('Found multiple channels. Process as RGB data? [Y/N]\n')
        if not rgb:
            hr = hr[0, 0, :, :, 0]
            lr = lr[0, 0, :, :, 0]
        else:
            hr = hr[0, 0, :, :, :]
            lr = lr[0, 0, :, :, :]
    else:
        raise Exception("Invalid file format -- use tif(f) or czi")

    factor = hr.shape[0] / lr.shape[0]

    if factor % 2 != 0:
        raise Exception("Unexpected image dimensions -- must differ by a power of 2")
    if int(math.log(factor, 2)) > 3:
        raise Exception("Unexpected image dimensions -- a resolution difference of 2, 4 and 8 is allowed")

    # Create paths
    Path(out_path, 'HR').mkdir(parents=True, exist_ok=True)
    Path(out_path, 'LR').mkdir(parents=True, exist_ok=True)
    if test_frac > 0:
        Path(out_path, 'TESTING', 'HR').mkdir(parents=True, exist_ok=True)
        Path(out_path, 'TESTING', 'LR').mkdir(parents=True, exist_ok=True)

    hr_tile_size = 256
    # Make sure padding is dividable by factor to avoid coordinate issues with down sampling/up scaling during template matching
    padding = int(math.ceil(hr_tile_size * padding_frac) - (
                math.ceil(hr_tile_size * padding_frac) % factor)) if padding_frac > 0 else 0

    if math.log(hr.shape[0] / lr.shape[0], 2) % 1 != 0 and math.log(hr.shape[1] / lr.shape[1], 2) % 1 != 0 and hr.shape[
        0] / lr.shape[0] != hr.shape[1] / lr.shape[1]:
        sys.exit('Image dimensionality problem -- check input files.')
    else:
        factor = int(hr.shape[0] / lr.shape[0])
        lr_tile_size = int(hr_tile_size / factor)

    n_x = int(hr.shape[0] / hr_tile_size)
    n_y = int(hr.shape[1] / hr_tile_size)
    if test_frac > 0:
        test_set = unique_tuples(n_x, n_y, n_x * n_y * test_frac)

    progress_bar = tqdm(desc='Creating image pairs', total=n_x * n_y)
    count_train = 0
    count_test = 0
    match_template_time = 0
    for i in range(n_x):
        for j in range(n_y):
            # Exclude edges
            if i not in [0, n_x - 1] and j not in [0, n_y - 1]:
                hr_tile = hr[i * hr_tile_size - padding:i * hr_tile_size + hr_tile_size + padding,
                          j * hr_tile_size - padding:j * hr_tile_size + hr_tile_size + padding]
                lr_tile = lr[i * lr_tile_size:i * lr_tile_size + lr_tile_size,
                          j * lr_tile_size:j * lr_tile_size + lr_tile_size]
                if padding_frac > 0:  # Accurate tile pairs using FFT template matching
                    coords, runtime = match(hr_tile, lr_tile, factor, False)
                    coords = tuple([int(factor * x) for x in coords])
                    hr_tile = hr_tile[coords[0]:coords[1], coords[2]:coords[3]]
                    match_template_time += runtime

                if hr_tile.shape != (hr_tile_size, hr_tile_size):
                    print('Incorrect tile shape. Skipping.')
                    continue

                if test_frac > 0 and (i, j) in test_set:
                    tifffile.imwrite(f'{out_path}/TESTING/HR/{name}_{i}_{j}.tif', hr_tile)
                    tifffile.imwrite(f'{out_path}/TESTING/LR/{name}_{i}_{j}.tif', lr_tile)
                    count_test += 1
                else:
                    tifffile.imwrite(f'{out_path}/HR/{name}_{i}_{j}.tif', hr_tile)
                    tifffile.imwrite(f'{out_path}/LR/{name}_{i}_{j}.tif', lr_tile)
                    count_train += 1
            progress_bar.update()
    progress_bar.close()
    return count_train, count_test
