import math
import random
import czifile
import tifffile
import numpy as np
from tqdm import tqdm
from pathlib import Path
from image_pairs import match
from scipy.ndimage import rotate

"""
In case of small datasets, this script can be used to augment the dataset to generate enough training data. 
"""


def rotate_point3(point, angle, origin):
    """
    Rotates a 2D point counterclockwise around a specified origin by a given angle (in radians).
    Uses trigonometric functions to calculate the new coordinates of the point.
    Returns the new coordinates as a list of integers.

    :param point: list
        List with X, Y coordinates of the point to be rotated
    :param angle: int
        The angle (in radians) by which to rotate the point
    :param origin: list
        List with X, Y coordinates that define the origin around which to rotate the point
    :return: list
        List with new X, Y coordinates
    """
    # Rotate a point counterclockwise by a given angle around a given origin. Angle in rad.
    x = origin[0] + math.cos(angle) * (point[0] - origin[0]) - math.sin(angle) * (point[1] - origin[1])
    y = origin[1] + math.sin(angle) * (point[0] - origin[0]) + math.cos(angle) * (point[1] - origin[1])
    return [int(x), int(y)]


def make_line(p1, p2):
    """
    Calculates the slope and intercept of a line passing through two points.

    :param p1: list
        List with X, Y coordinates of the first point
    :param p2: list
        List with X, Y coordinates of the second point
    :return: tuple with two floats
        Returns the slope and intercept of the line
    """
    d_y = p1[1] - p2[1]
    d_x = p1[0] - p2[0]
    slope = d_y / d_x if d_x != 0 else 0
    intercept = (p1[0] * p2[1] - p2[0] * p1[1]) / d_x if d_x != 0 else 0
    return slope, intercept


def augment_pair(hr_p,  # HR file path
                 lr_p,  # LR file path
                 name,  # Filename of map being processed
                 out_path,  # Output directory
                 n_angles=10,  # Number of random angles to sample from
                 n_lines=500,  # Number of random lines to generate over the image to sample from
                 sample=3,  # Number of random points to sample from each line
                 padding=200,  # Padding to prevent sampling outside the image
                 tile_padding_frac=0.1,  # Padding around tiles to allow for template matching, significantly improving
                 # pair alignment. To disable set to 0
                 test_frac=0  # Fraction of pairs reserved for testing/benchmarking. Data is not used during training
                 ):

    """
    The primary purpose of the augment_pair function is to generate and save augmented image tiles for training and testing machine learning models, ensuring accurate alignment between HR and LR image pairs.
    :param hr_p: string
        HR file path
    :param lr_p: string
        LR file path
    :param name: string
        name of the dataset
    :param out_path: string
        Output directory
    :param n_angles: int
        Number of random angles to sample from
    :param n_lines: int
        Number of random lines to generate over the image to sample from
    :param sample: int
        Number of random points to sample from each line
    :param padding: int
        Padding, in pixels, to prevent sampling outside the image
    :param tile_padding_frac: float (between 0 and 1)
        Padding around tiles to allow for template matching, which significantly improves pair alignment
    :param test_frac: float (between 0 and 1)
        Fraction of image pairs that are reserved for testing and benchmarking. This data will not be used during model training.
    :return: tuple
        Tuple with the number of training and test image pairs
    """
    # hr = np.random.randint(256, size=(4096, 2048), dtype='uint8')
    # lr = np.random.randint(256, size=(2048, 1024), dtype='uint8')

    hr_tile_size = 256

    count = 0
    count_train = 0
    count_test = 0

    if hr_p.lower().endswith(('.tif', '.tiff')):
        hr = tifffile.imread(hr_p)
        lr = tifffile.imread(lr_p)
        # Select first channel if more than 1
        if len(hr.shape) > 2:
            hr = hr[:, :, 0]
            lr = lr[:, :, 0]
    elif hr_p.lower().endswith('.czi'):
        hr = czifile.imread(hr_p)
        lr = czifile.imread(lr_p)
        # Extract relevant channel
        hr = hr[0, 0, :, :, 0]
        lr = lr[0, 0, :, :, 0]
    else:
        raise Exception("Invalid file format -- use tif(f) or czi")

    factor = int(hr.shape[0] / lr.shape[0])
    if factor % 2 != 0:
        raise Exception("Unexpected image dimensions -- must differ by a power of 2")
    if int(math.log(factor, 2)) > 3:
        raise Exception("Unexpected image dimensions -- a resolution difference of 2, 4 and 8 is allowed")

    # Create paths
    Path(out_path, 'HR').mkdir(parents=True, exist_ok=True)
    Path(out_path, 'LR').mkdir(parents=True, exist_ok=True)
    if test_frac > 0:
        Path(out_path, 'TESTING', 'HR').mkdir(parents=True, exist_ok=True)
        Path(out_path, 'Testing', 'LR').mkdir(parents=True, exist_ok=True)

    tiles_in_image = int(hr.shape[0] / hr_tile_size * hr.shape[1] / hr_tile_size)
    tiles_sampled = n_angles * n_lines * sample
    print(
        f'Sampling {tiles_sampled} augmented tiles, there are {tiles_in_image} tiles in image. Rotations take some time to process -- be patient.')

    if test_frac > 0:
        test_set = random.sample(range(0, tiles_sampled), int(tiles_sampled * test_frac))

    # Debugging
    mark_corner = False
    plot_lines = False

    # Corner coordinates with paddin
    tl = [padding, padding]
    tr = [padding, hr.shape[1] - padding]
    bl = [hr.shape[0] - padding, padding]
    br = [hr.shape[0] - padding, hr.shape[1] - padding]

    progress_bar = tqdm(desc='Creating image pairs', total=tiles_sampled)

    for i in range(n_angles):
        deg = np.random.randint(2, 90)
        rad = deg * math.pi / 180
        # print(f'Angle:{deg}')

        # Rotate using scipy.ndimage
        hr_rot = rotate(hr, deg, reshape=True)
        lr_rot = rotate(lr, deg, reshape=True)

        # Rotate corner coordinates around center
        tl_r = rotate_point3(tl, rad, [int(0.5 * hr.shape[0]), int(0.5 * hr.shape[1])])
        tr_r = rotate_point3(tr, rad, [int(0.5 * hr.shape[0]), int(0.5 * hr.shape[1])])
        bl_r = rotate_point3(bl, rad, [int(0.5 * hr.shape[0]), int(0.5 * hr.shape[1])])
        br_r = rotate_point3(br, rad, [int(0.5 * hr.shape[0]), int(0.5 * hr.shape[1])])

        # Correct for expansion of canvas to accommodate rotation
        add = [int(0.5 * (hr_rot.shape[0] - hr.shape[0])), int(0.5 * (hr_rot.shape[1] - hr.shape[1]))]
        for point in [tl_r, tr_r, bl_r, br_r]:
            point[0] += add[0]
            point[1] += add[1]

        # For debugging
        if mark_corner:
            hr_rot[tl_r[0] - 25:tl_r[0] + 25, tl_r[1]:tl_r[1] + 50] = 255
            hr_rot[tr_r[0]:tr_r[0] + 50, tr_r[1] - 25:tr_r[1] + 25] = 255
            hr_rot[bl_r[0] - 50:bl_r[0], bl_r[1] - 25:bl_r[1] + 25] = 255
            hr_rot[br_r[0] - 25:br_r[0] + 25, br_r[1] - 50:br_r[1]] = 255

        # Randomly sample tiles
        for j in range(n_lines):

            tl_tr_line = make_line(tl_r, tr_r)
            bl_br_line = make_line(bl_r, br_r)
            tl_bl_line = make_line(tl_r, bl_r)
            tr_br_line = make_line(tr_r, br_r)

            # Random points on opposite sides from random set of axes
            if np.random.randint(0, 2) == 0:
                x_1 = np.random.randint(high=tl_r[0], low=tr_r[0])
                x_2 = np.random.randint(high=bl_r[0], low=br_r[0])
                y_1 = int(tl_tr_line[0] * x_1 + tl_tr_line[1])
                y_2 = int(bl_br_line[0] * x_2 + bl_br_line[1])
            else:
                x_1 = np.random.randint(high=bl_r[0], low=tl_r[0])
                x_2 = np.random.randint(high=br_r[0], low=tr_r[0])
                y_1 = int(tl_bl_line[0] * x_1 + tl_bl_line[1])
                y_2 = int(tr_br_line[0] * x_2 + tr_br_line[1])

            if plot_lines:
                for k in range(tr_r[0], tl_r[0]):  # TL-TR line
                    y = int(tl_tr_line[0] * k + tl_tr_line[1])
                    hr_rot[k - 10:k + 10, y - 10:y + 10] = 255
                for k in range(br_r[0], bl_r[0]):  # BL-BR line
                    y = int(bl_br_line[0] * k + bl_br_line[1])
                    hr_rot[k - 10:k + 10, y - 10:y + 10] = 255
                for k in range(tl_r[0], bl_r[0]):  # TL-TR line
                    y = int(tl_bl_line[0] * k + tl_bl_line[1])
                    hr_rot[k - 10:k + 10, y - 10:y + 10] = 255
                for k in range(tr_r[0], br_r[0]):  # BL-BR line
                    y = int(tr_br_line[0] * k + tr_br_line[1])
                    hr_rot[k - 10:k + 10, y - 10:y + 10] = 255

            p1 = [x_1, y_1]
            p2 = [x_2, y_2]
            diagonal = make_line(p1, p2)

            tile_padding = int(math.ceil(hr_tile_size * tile_padding_frac) - (
                    math.ceil(hr_tile_size * tile_padding_frac) % factor)) if tile_padding_frac > 0 else 0

            match_template_time = 0
            for k in range(sample):
                # Random point on line
                x = np.random.randint(low=min(p1[0], p2[0]), high=max(p1[0], p2[0])) if p1[0] != p2[0] else p1[
                    0]  # If else to fix p1[0] = p2[0] edge case
                y = int(diagonal[0] * x + diagonal[1])

                # Plot areas (debugging visualization, uncomment last 2 lines as well)
                # hr_rot[x-128:x+128,y-128:y+128] = 255

                hr_crop = hr_rot[x - int(0.5 * hr_tile_size) - tile_padding:x + int(0.5 * hr_tile_size) + tile_padding,
                          y - int(0.5 * hr_tile_size) - tile_padding:y + int(0.5 * hr_tile_size) + tile_padding]
                lr_crop = lr_rot[int(x / factor) - int(0.5 * hr_tile_size / factor):int(x / factor) + int(
                    0.5 * hr_tile_size / factor),
                          int(y / factor) - int(0.5 * hr_tile_size / factor):int(y / factor) + int(
                              0.5 * hr_tile_size / factor)]
                if tile_padding_frac > 0:  # Accurate tile pairs using FFT template matching
                    coords, runtime = match(hr_crop, lr_crop, factor, False)
                    coords = tuple([int(factor * x) for x in coords])
                    hr_crop = hr_crop[coords[0]:coords[1], coords[2]:coords[3]]
                    match_template_time += runtime

                if hr_crop.shape != (hr_tile_size, hr_tile_size):
                    print('Incorrect tile shape. Skipping.')
                    continue

                if test_frac > 0 and count in test_set:
                    tifffile.imwrite(f'{out_path}/TESTING//HR/{name}_{i}_{j}_{k}.tif', hr_crop)
                    tifffile.imwrite(f'{out_path}/TESTING/LR/{name}_{i}_{j}_{k}.tif', lr_crop)
                    count_test += 1
                else:
                    tifffile.imwrite(f'{out_path}/HR/{name}_{i}_{j}_{k}.tif', hr_crop)
                    tifffile.imwrite(f'{out_path}/LR/{name}_{i}_{j}_{k}.tif', lr_crop)
                    count_train += 1
                count += 1
                progress_bar.update()
    # print(f'Template matching time: {match_template_time} S')
    progress_bar.close()
    # To plot sampled locations for each rotation angle. Change folder name below!
    # tifffile.imwrite(f'data/HR_rot_{i}.tif', hr_rot)
    # tifffile.imwrite(f'data/LR_rot_{i}.tif', lr_rot)

    return count_train, count_test
