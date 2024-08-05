import os
import gc
import time
import numpy as np
import scipy.ndimage
from tqdm import tqdm
import multiprocessing
from contextlib import contextmanager

nProcessesDefault = multiprocessing.cpu_count()


@contextmanager
def memmap_context(filename, dtype, mode, shape):
    """
    Isolate the file creation and usage entirely within a context manager as workaround for issues with deleting memmap
    temporary files.
    """
    mm = np.memmap(filename, dtype=dtype, mode=mode, shape=shape)
    try:
        yield mm
    finally:
        mm._mmap.close()
        del mm
        gc.collect()
        time.sleep(1)  # Wait for file handle release
        try:
            os.remove(filename)
            print(f"Temporary file {filename} successfully removed.")
        except PermissionError as e:
            print(f"Could not remove temporary file: {e}")


def applyPhiPython(im, Phi=None, PhiCentre=None, interpolationOrder=3, max_len=1000000):
    """
    Deform a 3D image using a deformation function "Phi", applied using scipy.ndimage.map_coordinates.
    :param im: 3D numpy array
        3D array to be deformed
    :param Phi: 4x4 array
        Deformation function Phi
    :param PhiCentre: 3x1 array (floats)
        Centre of application of Phi. Default is center of image.
    :param interpolationOrder: int
        Order of image interpolation to use.
    :param max_len: int
        Maximum length of tensor. Lower this value if running into memory issues.
    :return: 3D array
        Input deformed by Phi
    """

    print('Transforming data by applying Phi:')
    if Phi is None:
        PhiInv = np.eye(4, dtype='<f4')
    else:
        try:
            PhiInv = np.linalg.inv(Phi).astype('<f4')
        except np.linalg.LinAlgError:
            PhiInv = np.eye(4, dtype='<f4')

    if PhiCentre is None:
        PhiCentre = (np.array(im.shape) - 1) / 2.0

    # Initialize coordinates
    imDef = np.zeros_like(im, dtype='<f4')

    # Memory-mapped file setup
    n_voxels = im.shape[0] * im.shape[1] * im.shape[2]
    coordinatesInitial_file = 'coordinatesInitial.dat'

    mgrid_shape = (3,) + im.shape
    coordinates_memmap_file = 'coordinates_mgrid.dat'

    with memmap_context(coordinatesInitial_file, dtype='<f4', mode='w+', shape=(3, n_voxels)) as coordinatesInitial, \
            memmap_context(coordinates_memmap_file, dtype='<f4', mode='w+', shape=mgrid_shape) as coordinates_memmap:

        # Fill coordinates using mgrid and reshape
        coordinates_memmap[0, :, :, :] = np.arange(im.shape[0]).reshape((im.shape[0], 1, 1))
        coordinates_memmap[1, :, :, :] = np.arange(im.shape[1]).reshape((1, im.shape[1], 1))
        coordinates_memmap[2, :, :, :] = np.arange(im.shape[2]).reshape((1, 1, im.shape[2]))

        coordinatesInitial[0, :] = coordinates_memmap[0].ravel() - PhiCentre[0]
        coordinatesInitial[1, :] = coordinates_memmap[1].ravel() - PhiCentre[1]
        coordinatesInitial[2, :] = coordinates_memmap[2].ravel() - PhiCentre[2]

        print(f'Coordinates initialized with shape: {coordinatesInitial.shape}. Applying Phi to set of coordinates.')

        # Apply Phi to coordinates
        coordinatesDef = np.zeros_like(coordinatesInitial)
        for i in tqdm(range(0, coordinatesInitial.shape[1], max_len)):
            end = min(i + max_len, coordinatesInitial.shape[1])
            slice = coordinatesInitial[:, i:end]
            dot = np.dot(PhiInv[:3, :3], slice)
            coordinatesDef[:, i:end] = dot + PhiCentre[:, np.newaxis]

        print('Mapping coordinates to data -- for large datasets this can take a while')
        imDef += scipy.ndimage.map_coordinates(im, coordinatesDef, order=interpolationOrder).reshape(
            imDef.shape).astype(
            '<f4')
        print('Mapping completed: data successfully transformed.')

    return imDef
