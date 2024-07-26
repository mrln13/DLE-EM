import os
import sys
import json
import reg2D2D
import match2D
from utils import *
import tkinter as tk
from tqdm import tqdm
from datetime import *
from pathlib import Path


"""
Image registration for DLE-EM workflow. For details on called functions refer to associated python scripts. 
"""


def main():
    root = tk.Tk()
    root.withdraw()
    filetypes = [('tif files', '.tif'), ('tiff files', '.tiff'), ('czi files', '.czi')]

    map = None
    tile = None
    multi = False

    # Resolution difference. Specify, or set to None for prompt
    factor = None

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
            LR_tile = select_file(f"Select extracted LR tile {i + 1}", "Select LR tile:", filetypes=filetypes)

            # HR tile
            HR_tile = select_file(f"Select extracted HR tile {i + 1}", "Select HR tile:", filetypes=filetypes)
            extracted_tiles.append([LR_tile, HR_tile])

    # Difference in resolution
    if factor is None:
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
                                    binStart=16,  # Factor to start binning with
                                    binStop=1,  # Stop binning at
                                    multi_path=multi_path,
                                    plot=True)
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
            matched_path = match2D.match2D(files=pair, factor=factor)
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
