import numpy
import scipy.ndimage
from tqdm import tqdm
import multiprocessing

nProcessesDefault = multiprocessing.cpu_count()


def applyPhiPython(im, Phi=None, PhiCentre=None, interpolationOrder=3, max_len=10000000):

    """
    Deform a 2D or 3D image using a deformation function "Phi", applied using scipy.ndimage.map_coordinates
    Can have orders > 1 but is hungry in memory. Modified from SPAM package: https://pypi.org/project/spam/.
    :param im: 3D numpy array
        3D array that is to be deformed
    :param Phi:4x4 array
        Deformation function Phi
    :param PhiCentre: 3x1 array (floats)
        Centre of application of Phi. Default (numpy.array(im1.shape)-1)/2.0, i.e. centre of image.
    :param interpolationOrder: int
        Order of image interpolation to use. This value is passed directly to ``scipy.ndimage.map_coordinates`` as "order".
        Default = 3
    :param max_len: int
        Maximum length of tensor. Default value: 10000000. When running into memory issues, lower this value.
    :return: 3D array
        Input deformed by Phi
    """

    print('Transforming data by applying Phi:')
    if Phi is None:
        PhiInv = numpy.eye(4, dtype='<f4')
    else:
        try:
            PhiInv = numpy.linalg.inv(Phi).astype('<f4')
        except numpy.linalg.linalg.LinAlgError:
            # print( "\tapplyPhiPython(): Can't inverse Phi, setting it to identity matrix. Phi is:\n{}".format( Phi ) )
            PhiInv = numpy.eye(4)

    if PhiCentre is None:
        PhiCentre = (numpy.array(im.shape) - 1) / 2.0

    # Initialize coordinates
    imDef = numpy.zeros_like(im, dtype='<f4')
    coordinatesInitial = numpy.ones((4, im.shape[0] * im.shape[1] * im.shape[2]), dtype='<f4')
    coordinates_mgrid = numpy.mgrid[0:im.shape[0],
                                    0:im.shape[1],
                                    0:im.shape[2]]

    coordinatesInitial[0, :] = coordinates_mgrid[0].ravel() - PhiCentre[0]
    coordinatesInitial[1, :] = coordinates_mgrid[1].ravel() - PhiCentre[1]
    coordinatesInitial[2, :] = coordinates_mgrid[2].ravel() - PhiCentre[2]

    print(f'Coordinates initialized with shape: {coordinatesInitial.shape}. Applying Phi to set of coordinates. ')

    # Apply Phi to coordinates
    if coordinatesInitial.shape[1] > max_len:  # For large arrays calculate dot product in parts (numpy memory limitation)
        print('Array cannot be processed at once -- slicing array:')
        coordinatesDef = numpy.zeros_like(coordinatesInitial)
        for i in tqdm(range(int(coordinatesInitial.shape[1] / max_len) if coordinatesInitial.shape[1] % max_len == 0 else int(coordinatesInitial.shape[1] / max_len) + 1)):
            slice = coordinatesInitial[:, i * max_len:i * max_len + max_len]
            dot = numpy.dot(PhiInv, slice)
            coordinatesDef[:, i * max_len:i * max_len + dot.shape[1]] = dot

    else:
        coordinatesDef = numpy.dot(PhiInv, coordinatesInitial)

    coordinatesDef[0, :] += PhiCentre[0]
    coordinatesDef[1, :] += PhiCentre[1]
    coordinatesDef[2, :] += PhiCentre[2]

    print('Mapping coordinates to data -- for large datasets this can take a while')
    imDef += scipy.ndimage.map_coordinates(im,
                                           coordinatesDef[0:3],
                                           order=interpolationOrder).reshape(imDef.shape).astype('<f4')
    print('Mapping completed: data successfully transformed.')
    return imDef
