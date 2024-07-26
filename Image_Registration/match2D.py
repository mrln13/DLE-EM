import os
import applyPhi
import tifffile
import numpy as np
import matplotlib.pyplot as plt
import registerMultiscaleModified as reg
from skimage.transform import downscale_local_mean


def match2D(
        files,  # Paths to LR and HR files
        factor,  # Resolution difference factor
        binStart=16,  # Downscale factor to begin correlation with
        binStop=4,  # Final downscale factor
        maxIterations=250,  # Maximum iterations for each bin step)
        showProgress=True,
):
    """
    Perform multiscale subpixel image correlation between two images (HR and LR maps). The HR image will be corrected to match the LR image.
    For more details see registerModified and registerMultiscaleModified

    :param files: list of two strings
        List that contains the path to LR and HR files (in that order) that will be aligned
    :param factor: int
        Difference in resolution between HR and LR -- must be a power of 2
    :param binStart: int
        Start with downsampling by this factor -- must be a power of 2
    :param binStop: int
        Final binstep -- must be a power of 2
    :param maxIterations: int
        Maximum number of iterations performed for each(!) bin step
    :param showProgress: boolean
        Dispaly live progress of the image registration process
    :return: string
        Path of the corrected HR map
    """
    max_len = 1000000  # set max tensor size to avoid issues when calculating dot product. When running into memory
    # issues, lower this value

    # Read data
    tile_from_map = tifffile.imread(files[0])
    tile = tifffile.imread(files[1])

    # Validate input
    if tile_from_map.shape[0] * factor != tile.shape[0] or len(tile.shape) > 2:
        while True:
            inp = input(
                'Unexpected file dimensions. Are the input files reg2D2D output? Continue on own risk: [Y/N]?\n')
            if inp.lower() == 'y':
                break
            elif inp.lower() == 'n':
                raise Exception('Aborting.')
            print('Invalid input.')

    # Crop both tiles to same size
    crop = min(tile_from_map.shape[0], int(tile.shape[0] / factor)), min(tile_from_map.shape[1],
                                                                         int(tile.shape[1] / factor))
    tile = tile[:crop[0] * factor, :crop[1] * factor, ...]
    tile_from_map = tile_from_map[:crop[0], :crop[1], ...]
    # Rescale HR tile to match extracted tile shape
    tile_rescaled = downscale_local_mean(tile, (factor, factor)) if len(tile.shape) == 2 else downscale_local_mean(tile,
                                                                                                                   (
                                                                                                                       factor,
                                                                                                                       factor,
                                                                                                                       1))

    # Mask select regions (not) to correlate. 1: correlate, 0: not correlate
    mask = np.ones_like(tile) if len(tile.shape) == 2 else np.ones_like(tile[:, :, 0])
    plt.figure()
    result = reg.registerMultiscale(
        tile_from_map[:, :, 0] if len(tile_from_map.shape) == 3 else tile_from_map,
        tile_rescaled[:, :, 0] if len(tile_rescaled.shape) == 3 else tile_rescaled,
        # Rescale HR tile to match extracted tile shape
        im1mask=mask.astype('bool'),
        binStart=binStart,
        binStop=binStop,
        maxIterations=maxIterations,
        updateGradient=False,
        interpolator='C',
        verbose=True,
        imShowProgress=showProgress)

    plt.show()
    tile = np.expand_dims(tile, axis=0)

    print(f'Inverse deformation matrix:\n{result["PhiInv"]}')
    # Invert Phi - use obtained deformation matrix the other way around to deform HR tile to match LR map
    try:
        PhiInv = np.linalg.inv(result['Phi'].copy())
    except np.linalg.linalg.LinAlgError:
        PhiInv = np.eye(4)

    if len(tile.shape) == 4:
        corrected_tile = np.zeros_like(tile)
        for i in range(3):
            corrected = applyPhi.applyPhiPython(tile[:, :, :, i], Phi=result['PhiInv'],
                                                interpolationOrder=result['interpolationOrder'],
                                                max_len=max_len).astype('uint8')
            corrected_tile[:, :, :, i] = corrected
    else:
        corrected_tile = applyPhi.applyPhiPython(tile, Phi=result['PhiInv'],
                                                 interpolationOrder=result['interpolationOrder'],
                                                 max_len=max_len).astype('uint8')
    plt.close('all')
    folder = os.path.split(files[1])[0]
    path = os.path.join(folder, 'HR-corrected.tif')
    tifffile.imsave(path, corrected_tile[0])

    # Return path of output
    return path
