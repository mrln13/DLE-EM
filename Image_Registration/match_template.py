import numpy as np
from scipy.signal import fftconvolve
from collections.abc import Iterable

"""
Modified after scikit-image implementation of template matching 
"""


def check_nD(array, ndim, arg_name='image'):
    """
    Verify an array meets the desired ndims and array isn't empty.
    :param array: array-like
        Input array that is to be validated
    :param ndim: int (or iterable of ints)
        Allowable ndim or ndims for array
    :param arg_name: string
        The name of the array
    :return:
    """
    array = np.asanyarray(array)
    msg_incorrect_dim = "The parameter `%s` must be a %s-dimensional array"
    msg_empty_array = "The parameter `%s` cannot be an empty array"
    if isinstance(ndim, int):
        ndim = [ndim]
    if array.size == 0:
        raise ValueError(msg_empty_array % (arg_name))
    if array.ndim not in ndim:
        raise ValueError(
            msg_incorrect_dim % (arg_name, '-or-'.join([str(n) for n in ndim]))
        )


new_float_type = {
    # preserved types
    np.float32().dtype.char: np.float32,
    np.float64().dtype.char: np.float64,
    np.complex64().dtype.char: np.complex64,
    np.complex128().dtype.char: np.complex128,
    # altered types
    np.float16().dtype.char: np.float32,
    'g': np.float64,  # np.float128 ; doesn't exist on windows
    'G': np.complex128,  # np.complex256 ; doesn't exist on windows
}


def _supported_float_type(input_dtype, allow_complex=False):
    """Return an appropriate floating-point dtype for a given dtype.
    float32, float64, complex64, complex128 are preserved.
    float16 is promoted to float32.
    complex256 is demoted to complex128.
    Other types are cast to float64.
    Parameters
    ----------
    input_dtype : np.dtype or Iterable of np.dtype
        The input dtype. If a sequence of multiple dtypes is provided, each
        dtype is first converted to a supported floating point type and the
        final dtype is then determined by applying `np.result_type` on the
        sequence of supported floating point types.
    allow_complex : bool, optional
        If False, raise a ValueError on complex-valued inputs.
    Returns
    -------
    float_type : dtype
        Floating-point dtype for the image.
    """
    if isinstance(input_dtype, Iterable) and not isinstance(input_dtype, str):
        return np.result_type(*(_supported_float_type(d) for d in input_dtype))
    input_dtype = np.dtype(input_dtype)
    if not allow_complex and input_dtype.kind == 'c':
        raise ValueError("complex valued input is not supported")
    #    return new_float_type.get(input_dtype.char, np.float16)
    return new_float_type.get(input_dtype.char, np.float64)


def _window_sum_2d(image, window_shape):
    """
    Computation of the sum of elements in sliding windows of a specified shape over a 2D array.
    :param image: array-like
        Input image
    :param window_shape: tuple
        Specification of the sliding window
    :return: array-like
        Sum of elements for each window position
    """
    window_sum = np.cumsum(image, axis=0)
    window_sum = (window_sum[window_shape[0]:-1]
                  - window_sum[:-window_shape[0] - 1])

    window_sum = np.cumsum(window_sum, axis=1)
    window_sum = (window_sum[:, window_shape[1]:-1]
                  - window_sum[:, :-window_shape[1] - 1])

    return window_sum


def _window_sum_3d(image, window_shape):
    """
    Same as previous, but for 3D-data.
    :param image:
    :param window_shape:
    :return:
    """
    window_sum = _window_sum_2d(image, window_shape)

    window_sum = np.cumsum(window_sum, axis=2)
    window_sum = (window_sum[:, :, window_shape[2]:-1]
                  - window_sum[:, :, :-window_shape[2] - 1])

    return window_sum


# @profile
def match_template(image, template, pad_input=False, mode='constant',
                   constant_values=0):
    """
    Template matching in image by locating the regions in an image that match a given template by comparing the template to all possible positions in the image.
    :param image: array-like
        Larger image to locate the template within
    :param template: array-like
        Template to locate within the larger image
    :param pad_input: boolean
        Whether to pad the input image
    :param mode: string
        Padding mode (refer to numpy documentation)
    :param constant_values: int
        Constant values to use for padding if mode is 'constant' (refer to numpy documentation)
    :return: array
        Returns trimmed response array, containing normalized cross-correlation values that indicate how well the template
        matches different regions of the image
    """
    check_nD(image, (2, 3))

    if image.ndim < template.ndim:
        raise ValueError("Dimensionality of template must be less than or "
                         "equal to the dimensionality of image.")
    if np.any(np.less(image.shape, template.shape)):
        raise ValueError("Image must be larger than template.")

    image_shape = image.shape

    float_dtype = _supported_float_type(image.dtype)
    # image = image.astype(float_dtype, copy=False)

    pad_width = tuple((width, width) for width in template.shape)
    if mode == 'constant':
        image = np.pad(image, pad_width=pad_width, mode=mode,
                       constant_values=constant_values)
    else:
        image = np.pad(image, pad_width=pad_width, mode=mode)

    # Use special case for 2-D images for much better performance in computation of integral images
    if image.ndim == 2:
        image_window_sum = _window_sum_2d(image, template.shape)
        image_window_sum2 = _window_sum_2d(image ** 2, template.shape)
    elif image.ndim == 3:
        image_window_sum = _window_sum_3d(image, template.shape)
        image_window_sum2 = _window_sum_3d(image ** 2, template.shape)

    template_mean = template.mean()
    template_volume = np.prod(template.shape)
    template_ssd = np.sum((template - template_mean) ** 2)

    print("fft convolve starting")
    if image.ndim == 2:
        xcorr = fftconvolve(image, template[::-1, ::-1],
                            mode="valid")[1:-1, 1:-1]
    elif image.ndim == 3:
        xcorr = fftconvolve(image, template[::-1, ::-1, ::-1],
                            mode="valid")[1:-1, 1:-1, 1:-1]
    print("fft convolve finished")

    numerator = xcorr - image_window_sum * template_mean

    denominator = image_window_sum2
    np.multiply(image_window_sum, image_window_sum, out=image_window_sum)
    np.divide(image_window_sum, template_volume, out=image_window_sum, casting='unsafe')
    denominator -= image_window_sum
    denominator = denominator * template_ssd
    np.maximum(denominator, 0, out=denominator)  # sqrt of negative number not allowed
    np.sqrt(denominator, out=denominator)

    response = np.zeros_like(xcorr, dtype=float_dtype)

    # avoid zero-division
    mask = denominator > np.finfo(float_dtype).eps

    response[mask] = numerator[mask] / denominator[mask]

    slices = []
    for i in range(template.ndim):
        if pad_input:
            d0 = (template.shape[i] - 1) // 2
            d1 = d0 + image_shape[i]
        else:
            d0 = template.shape[i] - 1
            d1 = d0 + image_shape[i] - template.shape[i] + 1
        slices.append(slice(d0, d1))
    return response[tuple(slices)]
