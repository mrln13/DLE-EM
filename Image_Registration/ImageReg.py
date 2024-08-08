import os
import sys
import json
import reg2D2D
import match2D
import argparse
from utils import *
import tkinter as tk
from tqdm import tqdm
from datetime import *
from pathlib import Path


"""
Image registration for DLE-EM workflow. For details on called functions refer to associated python scripts. 
"""


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--factor", type=int, default=None, help="HR/LR disparity factor. Set to none for prompt.")
    parser.add_argument("--registration_binstart", type=int, default=32, help="Resize factor to start image registration binning with -- must be a power of 2")
    parser.add_argument("--registration_binstop", type=int, default=1, help="Resize factor at which to stop binning -- must be a power of 2")
    parser.add_argument("--correction_binstart", type=int, default=16, help="Resize factor to start correction of -- must be a power of 2")
    parser.add_argument("--correction_binstop", type=int, default=4, help="Resize factor at which to stop binning -- must be a power of 2")
    parser.add_argument("--plot", type=bool, default=True, help="Show live progress")
    parser.add_argument("--save_intermediate", type=bool, default=True, help="Save results for different bin steps")
    parser.add_argument("--log_2_crop", type=bool, default=True, help="Crop to a dimension that is a power of 2 (enable for best results)")
    parser.add_argument("--max_iterations", type=int, default=100, help="Maximum iterations for each bin step in the immage correction process")

    opt = parser.parse_args()
    print(opt)

    root = tk.Tk()
    root.withdraw()
    filetypes = [('czi files', '.czi'), ('tif files', '.tif'), ('tiff files', '.tiff')]

    map = None
    tile = None
    multi = False
    factor = opt.factor

    # Determine intended use
    use = int_inp("[1]: Locate HR tile(s) in LR map and extract files\n"
                  "[2]: Image registration of processed HR and LR tiles\n"
                  "[3]: Do both\n"
                  "[4]: Cancel\n", minval=1, maxval=4)

    if use == 4:
        sys.exit()

    if use in [1, 3]:
        if map is None:
            map = select_file('Select LR map.', 'Select LR map.', filetypes=filetypes)
            # One or more HR maps?
            multi = yes_no_inp('More than one HR region in the LR map [Y/N]?\n')

        # HR map
        if tile is None and not multi:
            tile = []
            file = select_file('Select HR map', 'Select HR map', filetypes=filetypes)
            tile.append(file)

        if tile is None and multi:
            tile = []
            hr_folder = select_folder('Select folder containing (only!) HR files.', 'Folder containing HR files:')
            for file in os.listdir(hr_folder):
                if file.lower().endswith(('czi', 'tif', 'tiff')):
                    tile.append(os.path.join(hr_folder, file))

    if use == 2:
        extracted_tiles = list()
        n_pairs = int_inp('Number of file pairs to process?\n')
        if n_pairs > 1:
            multi = True
        for i in range(n_pairs):
            # LR map
            LR_tile = select_file(f"Select extracted LR tile {i + 1}", "Select LR tile:", filetypes=[('tif files', '.tif'), ('tiff files', '.tiff')])

            # HR tile
            HR_tile = select_file(f"Select extracted HR tile {i + 1}", "Select HR tile:", filetypes=[('tif files', '.tif'), ('tiff files', '.tiff')])
            extracted_tiles.append([LR_tile, HR_tile])

    # Difference in resolution
    if factor is None or opt.factor not in [1, 2, 4, 8]:
        while True:
            factor = int_inp("Resolution difference factor [2/4/8].\n")
            if factor not in [1, 2, 4, 8]:
                print('Invalid input.')
                continue
            break

    if use in [1, 3]:
        '''
        Extracts location of the region in the LR map corresponding the HR map using fast Fourier transform (method modified 
        from scikit-image template matching. Optional arguments: plot (shows results, default: True), intermediate (save 
        intermediate results, default: True), log2crop (crop to dimension that is a power of 2 to improve results, default: 
        True, multi_path (used when processing multiple HR tiles, default: None, otherwise specify full path). 
        Output: HR tile and lR tile as single-channel tif-files.
        '''
        extracted_tiles = []
        now = datetime.now()
        # When processing multiple files, pass directory to reg2D
        multi_path = None if not multi else f'ImageReg-{now.strftime("%Y")}_{now.strftime("%m")}_{now.strftime("%d")}_{now.strftime("%H")}_{now.strftime("%M")}'
        for hr_file in tqdm(tile):
            print(f'\nCalling reg2D on: {hr_file}')
            results = reg2D2D.reg2D(map_path=map,
                                    tile_path=hr_file,
                                    factor=factor,  # Resolution difference factor between map and tile
                                    binStart=opt.registration_binstart,  # Factor to start binning with
                                    binStop=opt.registration_binstop,  # Stop binning at
                                    multi_path=multi_path,
                                    plot=opt.plot)
            print(f'Finished reg2D')
            extracted_tiles.append(results)
        if use == 3:
            final = yes_no_inp(
                'Continue using the results from the final bin step [Y] or manually select an intermediate result ['
                'N]?\n')
            if not final:
                for i in range(len(extracted_tiles)):
                    extracted_tiles[i][0] = select_file('Select (intermediate) LR tile.', 'Select LR tile:',
                                                        filetypes=filetypes)

    if use in [2, 3]:
        '''
        Registers HR map with corresponding region in the LR map by iteratively calculating a deformation matrix using 
        Optional arguments: binStart (factor to start binning with, default: 16), binStop (final bin factor, default: 8), 
        maxIterations (), showProgress ().
        '''
        for pair in extracted_tiles:
            print(f'Calling match2D')
            matched_path = match2D.match2D(files=pair,
                                           factor=factor,
                                           binStart=opt.correction_binstart,
                                           binStop=opt.correction_binstop,
                                           showProgress=opt.plot,
                                           maxIterations=opt.max_iterations)
            pair.append(matched_path)
            print(f'Finished match2D')

    # Convert list items to full paths and save to file
    for i in range(len(extracted_tiles)):
        for j in range(len(extracted_tiles[0])):
            extracted_tiles[i][j] = os.path.join(os.getcwd(), extracted_tiles[i][j])

    if use == 2 and multi:
        p = select_folder('Select location to save processed file list', 'Save file list in folder:')
    else:
        p = Path(extracted_tiles[0][0])
        if p.parts[-2] == 'intermediate':
            p = os.path.join(*(p.parts[:-2]))
        else:
            p = os.path.join(*(p.parts[:-1]))

    with open(f'{p}/ImageReg.txt', 'w') as fp:
        json.dump(extracted_tiles, fp)
    fp.close()


if __name__ == '__main__':
    main()
