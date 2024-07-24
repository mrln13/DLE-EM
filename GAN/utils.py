import os
import cv2
import numpy as np
from tkinter import filedialog
from tkinter import messagebox


"""
Basic functions used in multiple scripts
"""


def yes_no_inp(q):
    while True:
        a = input(q)
        if a.lower() == 'y':
            a = True
            break
        elif a.lower() == 'n':
            a = False
            break
        else:
            print('Invalid input.')
    return a


def int_inp(q, minval=0, maxval=10):
    while True:
        a = input(q)
        try:
            a = int(a)
        except ValueError:
            print('Input is not an integer.')
            continue
        if minval <= a <= maxval:
            break
        else:
            print(f'Input values between {minval} and {maxval} are accepted.')
    return a


def select_file(message, title, filetypes):
    messagebox.showinfo('Information', message)
    file_path = filedialog.askopenfilename(initialdir=os.getcwd(),
                                           title=title,
                                           filetypes=filetypes)
    if file_path is None or file_path == '':
        raise Exception('No file selected. Aborting.')
    return file_path


def select_folder(message, title):
    messagebox.showinfo('Information', message)
    folder_path = filedialog.askdirectory(initialdir=os.getcwd(),
                                          title=title)
    if folder_path is None or folder_path == '':
        raise Exception('No folder selected. Aborting')
    return folder_path


def unsharp_mask(image, kernel_size=(5, 5), sigma=1.0, amount=5.0, threshold=3):
    """Return a sharpened version of the image, using an unsharp mask."""
    blurred = cv2.GaussianBlur(image, kernel_size, sigma)
    sharpened = float(amount + 1) * image - float(amount) * blurred
    sharpened = np.maximum(sharpened, np.zeros(sharpened.shape))
    sharpened = np.minimum(sharpened, 255 * np.ones(sharpened.shape))
    sharpened = sharpened.round().astype(np.uint8)
    if threshold > 0:
        low_contrast_mask = np.absolute(image - blurred) < threshold
        np.copyto(sharpened, image, where=low_contrast_mask)
    return sharpened
