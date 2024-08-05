import os
import math
import czifile
import tifffile
import time as tm
import scipy.ndimage
from skimage import io
from datetime import *
from pathlib import Path
from match_template import *
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from skimage.transform import downscale_local_mean


# Crop from center
def crop_center(img, cropx, cropy):
    """
    Crops image around its center to a specified width and height
    :param img: array
        input to be cropped
    :param cropx: int
        width of the crop
    :param cropy: int
        height of the crop
    :return: array
        cropped image
    """
    y, x, *_ = img.shape
    startx = x // 2 - (cropx // 2)
    starty = y // 2 - (cropy // 2)
    return img[starty:starty + cropy, startx:startx + cropx, ...]


# Extracts tile from map that corresponds to input tile
def extract_from_map(map_, tile_, factor_, plot):
    """
    This function aims to find a small image (tile_) within a larger image (map_) and optionally plots
    the results. It can handle different magnifications by rescaling the images as needed.
    :param map_: array
        larger map where the search is performed
    :param tile_: array
        smaller image that is being searched for within map_
    :param factor_: int
        scaling factor to adjust for different resolutions between the map and the tile
    :param plot: boolean
        whether to plot the intermediate results
    :return: tupple, float
         coordinates of the bounding box of the matched region and the time taken for the matching process
    """
    #  Rescale in case of different magnification
    if factor_ != 1:
        if factor_ > 1:
            tile_ = downscale_local_mean(tile_, (factor_, factor_))
        if factor_ < 1:
            print(factor_)
            factor_ = 1 / factor_
            print(factor_)
            map_ = downscale_local_mean(map_, (factor_, factor_))
    start = tm.time()
    result = match_template(map_, tile_)
    end = tm.time()
    #  print('Match template time:', end - start)
    ij = np.unravel_index(np.argmax(result), result.shape)
    x, y = ij[::-1]
    htile, wtile = tile_.shape

    # Plot (intermediate) results
    if plot:
        fig = plt.figure(figsize=(8, 3))
        ax1 = plt.subplot(1, 3, 1)
        ax2 = plt.subplot(1, 3, 2)
        ax3 = plt.subplot(1, 3, 3, sharex=ax2, sharey=ax2)

        ax1.imshow(tile_, cmap=plt.cm.gray)
        ax1.set_axis_off()
        ax1.set_title('HR region')

        ax2.imshow(map_, cmap=plt.cm.gray)
        ax2.set_axis_off()
        ax2.set_title('LR map')
        # highlight matched region
        rect = plt.Rectangle((x, y), wtile, htile, edgecolor='r', facecolor='none')
        ax2.add_patch(rect)

        ax3.imshow(result)
        ax3.set_axis_off()
        ax3.set_title('Maximum')
        # highlight matched region
        ax3.autoscale(False)
        ax3.plot(x, y, 'o', markeredgecolor='r', markerfacecolor='none', markersize=10)

        plt.show(block=False)
        plt.pause(10)
        plt.close()

    coordinates = y, y + htile, x, x + wtile
    return coordinates, end - start


def reg2D(
        map_path,
        tile_path,
        factor,             # Resolution of tile with respect to map
        plot=True,          # Show results
        binStart=32,        # Resize factor to start binning with
        binStop=1,          # Resize factor to stop binning
        intermediate=True,  # Save intermediate results (in folder 'intermediate')
        log2crop=True,      # Crop to dimension that is a power of 2 (set to True for best results)
        multi_path=None     # Path when called to locate more than 1 HR region
):
    """
    Obtain location of template ("HR region") in map ("LR map") by bin-wise calling a modified version of scikit-image template matching.
    :param map_path: string
        path to map
    :param tile_path: string
        path to tile to locate within map
    :param factor: int
        factor of resolution difference
    :param plot: boolean
        show live results
    :param binStart: int
        resize factor to start binning with -- must be power of 2
    :param binStop: int
        resize factor at which to stop binning -- must be power of 2
    :param intermediate: boolean
        save intermediate results (in folder 'intermediate')
    :param log2crop: boolean
        crop to dimension that is a power of 2 (True for best results)
    :param multi_path: string
        Path that must be provided when locating more than one HR region within a LR manp
    :return: List with strings
        Paths to corresponding HR and LR maps

    """
    if map_path.lower().endswith(('.tif', '.tiff')):
        map = tifffile.imread(map_path)
        tile = tifffile.imread(tile_path)
    elif map_path.lower().endswith('.czi'):
        map = czifile.imread(map_path)
        tile = czifile.imread(tile_path)
        # Extract relevant channels
        map = map[0, 0, :, :, :]
        tile = tile[0, 0, :, :, :]
    else:
        raise Exception("Invalid file format -- use tif(f) or czi")

    to_single = False  # Take first channel when supplied with RGB data

    # Set and create output paths/folders
    now = datetime.now()
    if multi_path:
        output = os.path.join(multi_path, Path(tile_path).stem)
        Path(output).mkdir(parents=True, exist_ok=True)
        LR_path = os.path.join(output, 'LR-tile.tif')
        HR_path = os.path.join(output, 'HR-tile.tif')
        if intermediate:
            intermediate_path = os.path.join(output, 'intermediate')
            Path(intermediate_path).mkdir(parents=True, exist_ok=True)
    else:
        output = f'ImageReg-{now.strftime("%Y")}_{now.strftime("%m")}_{now.strftime("%d")}_{now.strftime("%H")}_{now.strftime("%M")}'
        Path(output).mkdir(parents=True, exist_ok=True)
        LR_path = os.path.join(output, 'LR-tile.tif')
        HR_path = os.path.join(output, 'HR-tile.tif')
        if intermediate:
            intermediate_path = os.path.join(output, 'intermediate')
            Path(intermediate_path).mkdir(parents=True, exist_ok=True)

    binLevels = 2 ** np.arange(math.log(binStart, 2), math.log(binStop, 2) - 1, -1).astype(
        int)  # Creates list of binlevels
    dim0 = 2 ** int(math.log(tile.shape[0], 2))  # Obtain closest log2 px dimensions
    dim1 = 2 ** int(math.log(tile.shape[1], 2))

    # Select first channel in case of RGB
    if to_single:
        if len(map.shape) == 3:
            map = map[:, :, 0]
        if len(tile.shape) == 3:
            tile = tile[:, :, 0]

    # Crop to pixel dimensions that are a power of 2
    if log2crop:
        tile = crop_center(tile, dim1, dim0)
    times = []  # Empty list for storing match_template running times

    # Work through binlevels
    for binLevel in binLevels:
        print(f'Working on binlevel {binLevel}')
        if len(map.shape) == 3:
            map = map[:,:,0]
        if len(tile.shape) == 3:
            tile = tile[:,:,0]
        map_binned = scipy.ndimage.zoom(map, 1 / binLevel, order=1)  # Provide only first channel
        tile_binned = scipy.ndimage.zoom(tile, 1 / binLevel, order=1)  # Same
        coordinates, runtime = extract_from_map(map_binned, tile_binned, factor, plot)  # Obtain region from map that matches input HR tile
        coordinates = tuple(element * binLevel for element in coordinates)
        times.append(runtime)  # Log time to estimate runtime for next iteration
        #  print(coordinates)
        if binLevel > binStop:
            x = np.array(list(range(np.where(binLevels == binLevel)[0][0] + 1)))
            y = np.array(times)
            y = np.log2(y)
            # Predict runtime for next iteration
            if np.where(binLevels == binLevel)[0][0] > 0:
                f = interp1d(x, y, fill_value='extrapolate')
                prediction = 2 ** f(np.where(binLevels == binLevel)[0][0] + 1)
                while True:
                    inp = input(
                        f'This iteration took {int(runtime)}S. Estimation of next iteration: {int(prediction)}S. Continue? [Y/N]\n')
                    if inp.lower() in ('y', 'n'):
                        break
                    print('Invalid input.')
            else:
                inp = 'y'

        if intermediate:
            tile_out = map[coordinates[0]:coordinates[1], coordinates[2]:coordinates[3]]
            io.imsave(os.path.join(intermediate_path, f'LR-tile_Bin_{binLevel}.tif'), tile_out)
        if binLevel == binStop or inp.lower() == 'n':
            tile_out = map[coordinates[0]:coordinates[1], coordinates[2]:coordinates[3]]
            io.imsave(LR_path, tile_out)
            io.imsave(HR_path, tile)
            break
    return [LR_path, HR_path]
