import numpy
import spam.DIC
import spam.label  # for im1mask
import spam.deformation
import spam.DIC.DICToolkit
from registerModified import register


def registerMultiscale(im1, im2, binStart, binStop=1, im1mask=None, PhiInit=None, PhiRigid=False, PhiInitBinRatio=1.0,
                       margin=None, maxIterations=100, deltaPhiMin=0.0001, updateGradient=False, interpolationOrder=1,
                       interpolator='C', verbose=False, imShowProgress=False, forceChangeScale=False):
    """
    Perform multiscale subpixel image correlation between im1 and im2. Modified from SPAM package: https://pypi.org/project/spam/.

    This means applying a downscale (binning) to the images, performing a Lucas and Kanade at that level,
    and then improving it on a 2* less downscaled image, all the way back to the full scale image.

    If your input images have multiple scales of texture, this should save significant time.

    Please see the documentation for `register` for more information.

    Parameters
    ----------
        im1 : 3D numpy array
            The greyscale image that will not move -- must not contain NaNs

        im2 : 3D numpy array
            The greyscale image that will be deformed -- must not contain NaNs

        binStart : int
            Maximum amount of binning to apply, please input a number which is 2^int

        binStop : int, optional
            Which binning level to stop upscaling at.
            The value of 1 (full image resolution) is almost always recommended (unless memory/time problems).
            Default = 1

        im1mask : 3D boolean numpy array, optional
            A mask for the zone to correlate in im1 with `False` in the zone to not correlate.
            Default = None, `i.e.`, correlate all of im1 minus the margin.
            If this is defined, the Phi returned is in the centre of mass of the mask

        PhiInit : 4x4 numpy array, optional
            Initial deformation to apply to im1, by default at bin1 scale
            Default = numpy.eye(4), `i.e.`, no transformation

        PhiRigid : bool, optional
            Run a rigid correlation? Only the rigid part of your PhiInit will be kept.
            Default = False

        PhiInitBinRatio : float, optional
            Change translations in PhiInit, if it's been calculated on a differently-binned image. Default = 1

        margin : int, optional
            Margin, in pixels, to take in im1.
            Can also be a N-component list of ints, representing the margin in ND.
            If im2 has the same size as im1 this is strictly necessary to allow space for interpolation and movement
            Default = 0 (`i.e.`, 10% of max dimension of im1)

        maxIterations : int, optional
            Maximum number of quasi-Newton iterations to perform before stopping. Default = 25

        deltaPhiMin : float, optional
            Smallest change in the norm of Phi (the transformation operator) before stopping. Default = 0.001

        updateGradient : bool, optional
            Should the gradient of the image be computed (and updated) on the deforming im2?
            Default = False (it is computed once on im1)

        interpolationOrder : int, optional
            Order of the greylevel interpolation for applying Phi to im1 when correlating. Recommended value is 3, but you can get away with 1 for faster calculations. Default = 3

        interpolator : string, optional
            Which interpolation function to use from `spam`.
            Default = 'python'. 'C' is also an option

        verbose : bool, optional
            Get to know what the function is really thinking, recommended for debugging only. Default = False

        imShowProgress : bool, optional
            Pop up a window showing a ``imShowProgress`` slice of the image differences (im1-im2) as im1 is progressively deformed.
            Default = False

        forceChangeScale : bool, optional
            Change up a scale even if not converged?
            Default = False

    Returns
    -------
        Dictionary:

            'Phi': 4x4 float array
                Deformation function defined at the centre of the image

            'returnStatus': signed int
                Return status from the correlation:

                2 : Achieved desired precision in the norm of delta Phi

                1 : Hit maximum number of iterations while iterating

                -1 : Error is more than 80% of previous error, we're probably diverging

                -2 : Singular matrix M cannot be inverted

                -3 : Displacement > 5*margin

            'error': float
                Error float describing mismatch between images, it's the sum of the squared difference divided by the sum of im1

            'iterations': int
                Number of iterations
    """
    # Detect unpadded 2D image first:
    if len(im1.shape) == 2:
        # pad them
        im1 = im1[numpy.newaxis, ...]
        im2 = im2[numpy.newaxis, ...]
        if im1mask is not None:
            im1mask = im1mask[numpy.newaxis, ...]

    # Detect 2D images
    if im1.shape[0] == 1:
        twoD = True
    else:
        twoD = False

    import math
    l = math.log(binStart, 2)
    if not l.is_integer():
        print("spam.DIC.correlate.registerMultiscale(): You asked for an initial binning of", binStart,
              ",rounding it to ", end='')
        binStart = 2 ** numpy.round(l)
        print(binStart)

    l = math.log(binStop, 2)
    if not l.is_integer():
        print("spam.DIC.correlate.registerMultiscale(): You asked for a final binning of", binStop, ",rounding it to ",
              end='')
        binStop = 2 ** numpy.round(l)
        print(binStop)

    # If there is no initial Phi, initalise it and im1defCrop to zero.
    if PhiInit is None:
        PhiInit = numpy.eye(4)
    else:
        # Apply binning on displacement   -- the /2 is to be able to *2 it in the LK call
        PhiInit[0:3, -1] *= PhiInitBinRatio / 2.0 / float(binStart)
    reg = {'Phi': PhiInit}

    if im1mask is not None:
        # Multiply up to 100 so we can apply a threshold below on binning in %
        im1mask = im1mask.astype('<u1') * 100

    # Generates a list of binning levels, if binStart=8 and binStop=2 this will be [8, 4 ,2]
    binLevels = 2 ** numpy.arange(math.log(binStart, 2), math.log(binStop, 2) - 1, -1).astype(int)
    for binLevel in binLevels:
        if verbose:
            print("Working on binning: ", binLevel)
        if binLevel > 1:
            if twoD:
                import scipy.ndimage
                im1b = scipy.ndimage.zoom(im1[0], 1 / binLevel, order=1)
                im2b = scipy.ndimage.zoom(im2[0], 1 / binLevel, order=1)
                # repad
                im1b = im1b[numpy.newaxis, ...]
                im2b = im2b[numpy.newaxis, ...]
                if im1mask is not None:
                    im1maskb = scipy.ndimage.zoom(im1mask[0], 1 / binLevel, order=1)
                    im1maskb = im1maskb[numpy.newaxis, ...]
                else:
                    im1maskb = None
            else:
                im1b = spam.DIC.binning(im1, binLevel)
                im2b = spam.DIC.binning(im2, binLevel)
                if im1mask is not None:
                    im1maskb = spam.DIC.binning(im1mask, binLevel) > 0
                else:
                    im1maskb = None
        else:
            im1b = im1
            im2b = im2
            if im1mask is not None:
                im1maskb = im1mask > 0
            else:
                im1maskb = None


        # Automatically calculate margin if none is passed
        # Detect default case and calculate margin necessary for a 45deg rotation with no displacement
        if margin is None:
            if twoD:
                # z-margin will be overwritten below
                marginB = [1 + int(0.1 * min(im1b.shape[1:]))] * 3
            else:
                marginB = [1 + int(0.1 * min(im1b.shape))] * 3


        elif type(margin) == list:
            marginB = (numpy.array(margin) // binLevel).tolist()

        else:
            # Make sure margin is an int
            margin = int(margin)
            margin = [margin] * 3
            marginB = (numpy.array(margin) // binLevel).tolist()

        reg = register(im1b, im2b,
                       im1mask=im1maskb,
                       PhiInit=reg['Phi'], PhiRigid=PhiRigid, PhiInitBinRatio=2.0,
                       margin=marginB,
                       maxIterations=maxIterations, deltaPhiMin=deltaPhiMin,
                       updateGradient=updateGradient,
                       interpolationOrder=interpolationOrder, interpolator=interpolator,
                       verbose=verbose,
                       imShowProgress=imShowProgress)

        if reg['returnStatus'] != 2 and not forceChangeScale:
            if verbose:
                print(
                    "spam.DIC.registerMultiscale(): binning {} did not converge (return Status = {}), not continuing".format(
                        binLevel, reg['returnStatus']))
                # Multiply up displacement and return bad result
            reg['Phi'][0:3, -1] *= float(binLevel)
            return reg

    # Rescale to full resolution
    reg['Phi'][0:3, -1] *= float(binStop)
    reg['PhiInv'][0:3, -1] *= float(binStop)

    return reg
