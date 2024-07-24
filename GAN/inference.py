import czifile
import warnings
import tifffile
import rasterio
from utils import *
from models import *
from tqdm import tqdm
from datasets import *
from pathlib import Path
from torch.autograd import Variable
from rasterio.windows import Window
from torch.utils.data import DataLoader
from rasterio.transform import from_origin
import torchvision.transforms as transforms
from torchvision.utils import save_image, make_grid
from rasterio.errors import NotGeoreferencedWarning


warnings.filterwarnings('ignore', category=NotGeoreferencedWarning)
Image.MAX_IMAGE_PIXELS = None


def main():
    """
    Function that handles upscaling low-resolution (LR) images or processing paired datasets for evaluating model performance using a pretrained generative model.
    """

    seamless = True  # No seam lines in up-scaled data -- comes with a minor computational expense

    weights = select_file('Select generator model.', 'Generator model.', [('PyTorch state dictionary', '.pth')])
    factor = int_inp('Factor of resolution difference?\n', minval=2, maxval=8)
    usage = int_inp('Upscale LR map [1] or process paired dataset (e.g. for evaluating model performance) [2]?\n',
                    minval=1, maxval=2)

    if usage == 2:
        separate = False  # Save original LR and HR images as well
        weights2 = select_file('Select second generator model.', 'Generator model', [('PyTorch state dictionary', '.pth')]) if yes_no_inp('Compare two models [y/n]?\n') else None
        datadir = select_folder('Select folder containing paired data.', 'Select folder')
        composite_grid = yes_no_inp('Create composite grids of the results?\n')

    else:
        tile_size = 750  # Choose appropriate tile size wrp to GPU memory
        weights2 = None
        composite_grid = False
        separate = False
        lr = select_file('Select LR map.', 'Select map',
                         [('tif files', '.tif'), ('tiff files', '.tiff'), ('czi files', '.czi')])
        upscaling_strat = int_inp(
            'Generate upscaled tiles based on the input map [0], generate an upscaled map [1], or do both [2]\n', 0, 2)

    output = select_folder('Select output folder.', 'Select folder')

    if usage == 2:
        # Create folders in output
        if separate:
            Path(os.path.join(output), 'LR').mkdir(parents=True, exist_ok=True)
            Path(os.path.join(output), 'HR').mkdir(parents=True, exist_ok=True)
        Path(os.path.join(output), 'GEN').mkdir(parents=True, exist_ok=True)
        if weights2 is not None:
            Path(os.path.join(output), 'GEN2').mkdir(parents=True, exist_ok=True)
        if composite_grid:
            Path(os.path.join(output), 'grids').mkdir(parents=True, exist_ok=True)
        # Use dataloader to generate batches
        dataloader = DataLoader(
            PairedImageDataset(os.path.join(datadir, "HR"), os.path.join(datadir, "LR")),
            batch_size=50,  # Set appropriate for gpu memory
            shuffle=False,
            num_workers=0,
        )
        n_batches = int_inp(f'Number of batches to run inference on? Dataloader contains {len(dataloader)} batches. \n', minval=1, maxval=len(dataloader))

    # For MSE loss function on single channel images
    mean = 0.5
    std = 0.5

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])

    cuda = torch.cuda.is_available()
    Tensor = torch.cuda.FloatTensor if cuda else torch.Tensor

    with torch.no_grad():
        generator = GeneratorResNet(factor=factor)
        generator.load_state_dict(torch.load(weights), strict=False)

        if cuda:
            generator.cuda()

        generator.eval()  # Set to evaluation mode

        if weights2 is not None:
            generator2 = GeneratorResNet(factor=factor)
            generator2.load_state_dict(torch.load(weights2), strict=False)
            if cuda:
                generator2.cuda()
            generator2.eval()

        if usage == 2:  # Process paired dataset
            count = 0
            for i, imgs in tqdm(enumerate(dataloader), total=n_batches):
                if count == n_batches:
                    break
                imgs_lr = Variable(imgs["lr"].type(Tensor))
                imgs_hr = Variable(imgs["hr"].type(Tensor))
                gen_hr = generator(imgs_lr)
                if weights2 is not None:
                    gen_hr2 = generator2(imgs_lr)

                for j in range(gen_hr.shape[0]):
                    if separate:
                        im_lr = imgs_lr[j, 0, :, :]
                        save_image(im_lr, os.path.join(output, 'LR', f"{i}_{j}.tif"), normalize=True)
                        im_hr = imgs_hr[j, 0, :, :]
                        save_image(im_hr, os.path.join(output, 'HR', f"{i}_{j}.tif"), normalize=True)
                    im_gen = gen_hr[j, 0, :, :]
                    save_image(im_gen, os.path.join(output, 'GEN', f"{i}_{j}.tif"), normalize=True)

                    if weights2 is not None:
                        im_gen2 = gen_hr2[j, 0, :, :]
                        save_image(im_gen2, os.path.join(output, 'GEN2', f"{i}_{j}.tif"), normalize=True)

                imgs_lr = nn.functional.interpolate(imgs_lr, scale_factor=factor)
                imgs_hr = Variable(imgs["hr"].type(Tensor))

                if composite_grid:
                    imgs_lr = make_grid(imgs_lr, nrow=1, normalize=True)
                    imgs_hr = make_grid(imgs_hr, nrow=1, normalize=True)
                    gen_hr = make_grid(gen_hr, nrow=1, normalize=True)
                    if weights2 is not None:
                        gen_hr2 = make_grid(gen_hr2, nrow=1, normalize=True)

                    if weights2 is None:
                        img_grid = torch.cat((imgs_lr, gen_hr, imgs_hr), -1)
                    else:
                        img_grid = torch.cat((imgs_lr, gen_hr, gen_hr2, imgs_hr), -1)

                    save_image(img_grid, os.path.join(output, f"grids/{i}.tif"), normalize=True)
                count += 1

        else:  # Process map
            if lr.lower().endswith('.czi'):
                img = czifile.imread(lr)
                img = img[0, 0, :, :, 0]
            else:
                img = tifffile.imread(lr)

            print("Input image dimensions:", img.shape)
            y_tiles = int(img.shape[0] / tile_size) + 1 if img.shape[0] % tile_size != 0 else int(
                img.shape[0] / tile_size)
            x_tiles = int(img.shape[1] / tile_size) + 1 if img.shape[1] % tile_size != 0 else int(
                img.shape[1] / tile_size)
            print(f"Splitting data in {y_tiles} by {x_tiles} tiles.")

            output_path = os.path.join(output, f'{Path(lr).stem}_upscaled.tif')

            padding = int(tile_size * 0.01) if seamless else 0  # Add overlap for seamless inference
            progress_bar = tqdm(desc='Processing data:', total=y_tiles * x_tiles)

            # Define transform for the entire image
            rastio_transform = from_origin(0, 0, 1, 1)
            out_tile_size = tile_size * factor

            # If tiles create separate folder
            if upscaling_strat in (0, 2):
                Path(os.path.join(output), 'Tiles').mkdir(parents=True, exist_ok=True)

            with rasterio.open(
                    output_path, 'w', driver='GTiff', height=out_tile_size * y_tiles, width=out_tile_size * x_tiles,
                    count=1, dtype=np.uint8, bigtiff='YES', transform=rastio_transform) as dst:
                for i in range(y_tiles):
                    for j in range(x_tiles):
                        tile = img[0 if i == 0 else i * tile_size - padding: i * tile_size + tile_size + padding,
                               0 if j == 0 else j * tile_size - padding: j * tile_size + tile_size + padding]
                        tile = transform(tile)
                        tile = tile.unsqueeze(0)
                        tile = tile.type(Tensor)
                        upscaled_tile = generator(tile)
                        upscaled_tile = upscaled_tile[:, :, 0 if i == 0 else padding * factor:-padding * factor,
                                        0 if j == 0 else padding * factor:-padding * factor]

                        out_tile = upscaled_tile.detach().cpu().numpy()
                        out_tile = ((out_tile + 1) / 2) * 255  # Map values from [-1, 1] to [0, 255]
                        out_tile = out_tile[0, 0, :, :].astype('uint8')
                        if upscaling_strat in (0, 2):
                            out_tile_path = os.path.join(output, 'Tiles', f"{Path(lr).stem}_upscaled_{i}_{j}.tif")
                            tifffile.imwrite(out_tile_path, out_tile)
                        if upscaling_strat in (1, 2):
                            window = Window(j * out_tile_size, i * out_tile_size, out_tile_size, out_tile_size)
                            dst.write(out_tile, window=window, indexes=1)
                        progress_bar.update()


if __name__ == '__main__':
    main()
