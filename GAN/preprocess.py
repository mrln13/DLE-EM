import json
import augment
from utils import *
import tkinter as tk
from pathlib import Path
from image_pairs import create_pairs

"""
Preprocess matching HR/LR dataset into (sub)pixel accurate HR/LR image pairs for GAN training. Creates folders for HR and LR images.
For more details on the preprocessing refer to the referenced python scripts. 
"""


def main():
    """
    Pre-processing data for input in SRGAN. When augmenting data, use only files that have been processed using ImageReg.
    When just creating image pairs, without augmenting, registered images obtained from a different source can be used.
    Corrected for resolution differences, input dimensions of HR and LR should be identical.
    """
    root = tk.Tk()
    root.withdraw()
    filetypes = [('tif files', '.tif'), ('tiff files', '.tiff'), ('czi files', '.czi')]  # Filetypes allowed

    paired_files = list()

    image_reg = yes_no_inp('Using ImageReg processed files? If so [Y], other registered image pairs [N].\n')

    # Load imageReg input
    if image_reg:
        input_file = select_file('Select ImageReg.txt file.', 'Select file:', [('txt files', '.txt')])
        with open(input_file) as fp:
            paired_files = json.load(fp)
        fp.close()
    # Other input files
    else:
        n_pairs = int_inp('Number of registered image pairs to process?\n')
        for i in range(n_pairs):
            # LR map
            LR_tile = select_file(f'Select LR tile{i + 1}', 'Select LR tile:', filetypes)
            HR_tile = select_file(f'Select HR tile{i + 1}', 'Select HR tile:', filetypes)
            paired_files.append([LR_tile, '', HR_tile])

    # Reserve percentage for testing? This data will not be used during training or validation. E.g. for benchmarking performance
    test_frac = 0.01 * int_inp(
        'Percentage to keep separate for testing? This data will not(!) be used during either training or '
        'validation.\n', minval=0, maxval=100)

    # To augment or not
    aug = yes_no_inp('Do you want to augment the data [Y/N]?\n')
    out_path = select_folder('Select folder to save preprocessing output.', 'Select output folder:')

    count_train = 0
    count_test = 0

    # Augmenting
    if aug:
        default = yes_no_inp('Use default augmenting parameters [Y/N]?\n')
        i = 0
        for pair in paired_files:
            i += 1
            print(f'Processing pair {i} from {len(paired_files)}.')
            file_p = Path(pair[0])
            file = file_p.parts[-2]
            # Default params
            if default:
                counts = augment.augment_pair(hr_p=pair[2],
                                              lr_p=pair[0],
                                              name=file,
                                              out_path=out_path,
                                              test_frac=test_frac)

            # Specify params
            else:
                print('Rotating large matrices is computational expensive, so sample multiple times per rotation to '
                      'generate sufficient data by specifying: the number of angles to sample from, the number of '
                      'random lines to generate over the image, and the number of samples to obtain per line.\n')
                n_angles = int_inp('Number of random angles to sample from:\n', maxval=1000)
                n_lines = int_inp('Number of random lines to generate over image \n', maxval=1000)
                sample = int_inp('Number of samples to obtain per line\n', maxval=1000)
                tile_padding_frac = 0.01 * int_inp('Percentage of tile padding to improve pair alignment? Default: '
                                                   '10. Set to 0 to disable.\n', minval=0, maxval=25)
                counts = augment.augment_pair(hr_p=pair[2],
                                              lr_p=pair[0],
                                              name=file,
                                              out_path=out_path,
                                              n_angles=n_angles,
                                              n_lines=n_lines,
                                              sample=sample,
                                              test_frac=test_frac,
                                              tile_padding_frac=tile_padding_frac)
            count_train += counts[0]
            count_test += counts[1]
    else:
        i = 0
        for pair in paired_files:
            i += 1
            print(f'Processing pair {i} from {len(paired_files)}.')
            file_p = Path(pair[0])
            file = file_p.parts[-2]
            counts = create_pairs(hr_p=pair[2],
                                  lr_p=pair[0],
                                  name=file,
                                  out_path=out_path,
                                  test_frac=test_frac)
            count_train += counts[0]
            count_test += counts[1]
    print(f'Done. {count_train} training and {count_test} testing image pairs were created.')


if __name__ == '__main__':
    main()
