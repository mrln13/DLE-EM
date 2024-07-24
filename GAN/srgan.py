import sys
import argparse
import pandas as pd
from utils import *
from models import *
import torch.nn as nn
from datasets import *
from datetime import *
import matplotlib.pyplot as plt
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision.utils import save_image, make_grid

"""
This script is designed to train a Generative Adversarial Network (GAN) for (single) image super-resolution. 
The GAN consists of a generator that upscales low-resolution images to high-resolution images and a discriminator that distinguishes between real high-resolution images and those generated by the generator.

Key Features:
Command-Line Arguments: The script accepts various command-line arguments to configure the training process, including the number of epochs, batch size, learning rate, and more.
Multi-GPU Support: If multiple GPUs are available, the script can leverage them for parallel processing.
Loading and Saving Models: The script can resume training from a specified epoch and saves model checkpoints at specified intervals.
Training and Validation: Supports splitting the dataset into training and validation sets and logs the loss for both.
Loss Functions: The script uses a combination of adversarial loss and content loss for training the generator.
Visualization and Logging: It can save image samples during training and log the loss values to a file. Optionally, it can plot the loss in real-time.

How to run: 
When the script is called, it asks for a folder containing a paired HR/LR training dataset. Then model training is initiated with the default parameters. 

Global workflow description:
Argument Parsing: Parses command-line arguments to configure the training.
Dataset Preparation: Loads the dataset and prepares DataLoader objects for batch processing.
Model Initialization: Initializes the generator, discriminator, and feature extractor models.
Training Loop: Iterates over the dataset, updating the generator and discriminator, and logs the training progress.
Validation Loop (optional): Evaluates the model on a validation set and logs the validation loss.
Checkpointing: Saves model weights at regular intervals.
To run the script, users need to provide appropriate arguments for their specific use case, such as dataset location, number of epochs, batch size, etc. 

Output:
The script creates two folders with a datetime-stamp: 
 1 - for saving the trained generator and discriminator models, and
 2 - images created by the generator are saved for evaluating the training process 

After training is finished, or stopped, training can be resumed by providing the --epoch=n argument, where n is the epoch from which training should be resumed.
"""

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epoch", type=int, default=0, help="epoch to start training from")
    parser.add_argument("--n_epochs", type=int, default=20, help="number of epochs of training")
    parser.add_argument("--dataset_name", type=str, default="Dataset", help="name of the dataset")
    parser.add_argument("--batch_size", type=int, default=80, help="size of the batches - depends on GPU memory")
    parser.add_argument("--lr", type=float, default=1e-4, help="adam: learning rate")
    parser.add_argument("--b1", type=float, default=0.5, help="adam: decay of first order momentum of gradient")
    parser.add_argument("--b2", type=float, default=0.999, help="adam: decay of first order momentum of gradient")
    parser.add_argument("--decay_epoch", type=int, default=5, help="epoch from which to start lr decay")
    parser.add_argument("--n_cpu", type=int, default=10, help="number of cpu threads to use during batch generation")
    parser.add_argument("--hr_height", type=int, default=256, help="high res. image height")
    parser.add_argument("--hr_width", type=int, default=256, help="high res. image width")
    parser.add_argument("--factor", type=int, default=4, help="Upscaling factor, must be exponential of 2")
    parser.add_argument("--channels", type=int, default=1, help="number of image channels")
    parser.add_argument("--sample_interval", type=int, default=50, help="interval between saving image samples")
    parser.add_argument("--checkpoint_interval", type=int, default=1, help="interval between model checkpoints")
    parser.add_argument("--parallel", type=int, default=1, help="enable multi GPU processing")
    parser.add_argument("--paired", type=int, default=1,
                        help="paired or unpaired dataset. Use 0 for artificial downsampling")
    parser.add_argument("--val_frac", type=float, default=0,
                        help="fraction of data to use for validation. 0 to disable")
    parser.add_argument("--plot_loss", type=int, default=0, help="plot loss during training")
    parser.add_argument("--log_loss", type=int, default=1, help="log training and validation loss in loss.txt")

    opt = parser.parse_args()
    print(opt)

    datadir = select_folder('Select folder containing paired dataset.', 'Select input folder:')
    train = True

    cuda = torch.cuda.is_available()

    hr_shape = (opt.hr_height, opt.hr_width)

    # Initialize generator and discriminator
    generator = GeneratorResNet(factor=opt.factor, in_channels=opt.channels, out_channels=opt.channels)
    discriminator = Discriminator(input_shape=(opt.channels, *hr_shape))
    feature_extractor = FeatureExtractor()

    # Set feature extractor to inference mode
    feature_extractor.eval()

    # Losses
    criterion_GAN = torch.nn.BCEWithLogitsLoss()  # torch.nn.MSELoss()
    temp_err = torch.nn.L1Loss()  # Change back to torch.nn.MSELoss()
    criterion_content = torch.nn.L1Loss()

    if torch.cuda.device_count() > 1 and opt.parallel == 1:
        print("Multiple GPU training")
        temp_err = nn.DataParallel(temp_err)
        generator = nn.DataParallel(generator)
        discriminator = nn.DataParallel(discriminator)
        feature_extractor = nn.DataParallel(feature_extractor)
        criterion_GAN = nn.DataParallel(criterion_GAN)
        criterion_content = nn.DataParallel(criterion_content)

    if cuda:
        temp_err = temp_err.cuda()
        generator = generator.cuda()
        discriminator = discriminator.cuda()
        feature_extractor = feature_extractor.cuda()
        criterion_GAN = criterion_GAN.cuda()
        criterion_content = criterion_content.cuda()

    if train:
        if opt.epoch != 0:
            inp = input("Saved models folder: ")
            timestamp = inp.split("_s")[0]
            saved_models = os.path.join(os.getcwd(), inp)
            imgs_out = os.path.join(os.getcwd(), timestamp + "_images")
            # Load pretrained models
            generator.load_state_dict(torch.load(f"{saved_models}/generator_%d.pth" % opt.epoch))
            discriminator.load_state_dict(torch.load(f"{saved_models}/discriminator_%d.pth" % opt.epoch))
            # done_imgs = []
            # for img in os.listdir(imgs_out):
            #     done_imgs.append(int(img.split(".")[0]))
            # addition = max(done_imgs)

            arg_dict = pd.read_csv(f"{saved_models}/args.txt", header=None, index_col=0, sep=";").squeeze(
                "columns").to_dict()

            if int(arg_dict["n_epochs"]) > opt.n_epochs or opt.epoch >= opt.n_epochs:
                opt.n_epochs = int(input("Total number of epochs to train:"))

            # Make sure essential parameters are consistent when resuming training
            opt.hr_height = int(arg_dict["hr_height"])
            opt.hr_width = int(arg_dict["hr_width"])
            opt.factor = int(arg_dict["factor"])
            opt.channels = int(arg_dict["channels"])
            opt.lr = float(arg_dict["lr"])
            opt.b1 = float(arg_dict["b1"])
            opt.b2 = float(arg_dict["b2"])
            opt.decay_epoch = int(arg_dict["decay_epoch"])
            opt.paired = int(arg_dict["paired"])
            opt.val_frac = float(arg_dict["val_frac"])
            opt.log_loss = int(arg_dict["log_loss"])

        else:
            now = datetime.now()
            imgs_out = f"{now.year}_{now.month}_{now.day}_{now.hour}_{now.minute}_images"
            saved_models = f"{now.year}_{now.month}_{now.day}_{now.hour}_{now.minute}_saved_models"
            os.makedirs(imgs_out, exist_ok=False)
            os.makedirs(saved_models, exist_ok=False)

            # Write arguments to file
            with open(f"{saved_models}/args.txt", "w") as file:
                for arg in vars(opt):
                    file.write(f"{arg};{getattr(opt, arg)}\n")
            # Create file for loss logging
            if opt.log_loss == 1:
                with open(f'{saved_models}/loss.txt', 'w') as f:
                    f.write('G_loss_train;D_loss_train;G_loss_val;D_loss_val\n') if opt.val_frac > 0.0 else f.write(
                        'G_loss_train;D_loss_train\n')

    # Optimizers
    optimizer_G = torch.optim.Adam(generator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))
    optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))

    Tensor = torch.cuda.FloatTensor if cuda else torch.Tensor

    # Initialize dataset
    if opt.paired == 0:
        dataset = ImageDataset(datadir, hr_shape=hr_shape, factor=opt.factor)
    else:
        dataset = PairedImageDataset(os.path.join(datadir, "HR"), os.path.join(datadir, "LR"))

    n_val = int(len(dataset) * opt.val_frac)  # Validation data
    n_train = len(dataset) - n_val  # Training data

    if opt.val_frac > 0.0:
        train_set, val_set = torch.utils.data.random_split(dataset, [n_train, n_val])  # Randomly split into subsets
    else:
        train_set = dataset

    train_loader = DataLoader(
        train_set,
        batch_size=opt.batch_size,
        shuffle=True,
        num_workers=opt.n_cpu,
    )
    print(f'Length training dataloader: {len(train_loader)}')

    if opt.val_frac > 0.0:
        val_loader = DataLoader(
            val_set,
            batch_size=opt.batch_size,
            shuffle=True,
            num_workers=opt.n_cpu,
        )
        print(f'Length validation dataloader: {len(val_loader)}')

    # Track loss stats
    loss_g_train = []
    loss_d_train = []
    if opt.val_frac > 0.0:
        loss_g_val = []
        loss_d_val = []

    # ----------
    #  Training
    # ----------
    begin_time = datetime.now()
    if train:
        if opt.plot_loss == 1:
            plt.ion()
            fig = plt.figure()
            ax = fig.add_subplot(111)

        min_val_loss = np.inf
        for epoch in range(opt.epoch, opt.n_epochs):
            epoch_loss_g_train = 0.0
            epoch_loss_d_train = 0.0
            for i, imgs in enumerate(train_loader):
                # Configure model input
                imgs_lr = Variable(imgs["lr"].type(Tensor))
                imgs_hr = Variable(imgs["hr"].type(Tensor))
                # Adversarial ground truths
                if torch.cuda.device_count() > 1 and opt.parallel == 1:
                    valid = Variable(Tensor(np.ones((imgs_lr.size(0), *discriminator.module.output_shape))),
                                     requires_grad=False)
                    fake = Variable(Tensor(np.zeros((imgs_lr.size(0), *discriminator.module.output_shape))),
                                    requires_grad=False)
                else:
                    valid = Variable(Tensor(np.ones((imgs_lr.size(0), *discriminator.output_shape))),
                                     requires_grad=False)
                    fake = Variable(Tensor(np.zeros((imgs_lr.size(0), *discriminator.output_shape))),
                                    requires_grad=False)
                # ------------------
                #  Train Generators
                # ------------------

                optimizer_G.zero_grad()

                # Generate a high resolution image from low resolution input
                gen_hr = generator(imgs_lr)

                # Adversarial loss
                loss_GAN = criterion_GAN(discriminator(gen_hr), valid)

                # Content loss
                if opt.channels == 1:  # Duplicate single channel to 3 channel tensor to match vgg19's expected input
                    fe_gen_hr = gen_hr.expand(-1, 3, -1, -1)
                    fe_imgs_hr = imgs_hr.expand(-1, 3, -1, -1)
                else:
                    fe_gen_hr = gen_hr
                    fe_imgs_hr = imgs_hr

                gen_features = feature_extractor(fe_gen_hr)
                real_features = feature_extractor(fe_imgs_hr)
                loss_content = temp_err(gen_hr, imgs_hr)

                # Total loss
                loss_G = loss_content + 1e-3 * loss_GAN

                if torch.cuda.device_count() > 1:
                    loss_G.sum().backward()
                else:
                    loss_G.backward()

                optimizer_G.step()

                # ---------------------
                #  Train Discriminator
                # ---------------------

                optimizer_D.zero_grad()

                # Loss of real and generated images
                loss_real = criterion_GAN(discriminator(imgs_hr), torch.ones_like(discriminator(imgs_hr)))
                loss_fake = criterion_GAN(discriminator(gen_hr.detach()),
                                          torch.zeros_like(discriminator(gen_hr.detach())))

                # Total loss
                loss_D = (loss_real + loss_fake) / 2
                if torch.cuda.device_count() > 1 and opt.parallel == 1:
                    loss_D.sum().backward()
                else:
                    loss_D.backward()

                optimizer_D.step()

                # --------------
                #  Log Progress
                # --------------
                epoch_loss_g_train += loss_G.mean().item()
                epoch_loss_d_train += loss_D.mean().item()
                sys.stdout.write(
                    "[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f]\n"
                    % (epoch, opt.n_epochs, i, len(train_loader), loss_D.mean().item(), loss_G.mean().item())
                )

                batches_done = epoch * len(train_loader) + i
                if batches_done % opt.sample_interval == 0:
                    # Save image grid with upsampled inputs, SRGAN outputs and ground truth
                    imgs_lr = nn.functional.interpolate(imgs_lr, scale_factor=opt.factor)
                    gen_hr = make_grid(gen_hr, nrow=1, normalize=True)
                    imgs_lr = make_grid(imgs_lr, nrow=1, normalize=True)
                    imgs_hr = make_grid(imgs_hr, nrow=1, normalize=True)
                    img_grid = torch.cat((imgs_lr, gen_hr, imgs_hr), -1)
                    save_image(img_grid, f"{imgs_out}/{epoch}_{i}.png", normalize=False)

            # Normalize losses and append to lists
            loss_g_train_norm = epoch_loss_g_train / len(train_loader)
            loss_d_train_norm = epoch_loss_d_train / len(train_loader)
            loss_g_train.append(loss_g_train_norm)
            loss_d_train.append(loss_d_train_norm)

            # --------------
            #  Validation
            # --------------

            if opt.val_frac > 0.0:
                epoch_loss_g_val = 0.0
                epoch_loss_d_val = 0.0

                # Set nets to evaluation
                generator.eval()
                discriminator.eval()
                with torch.no_grad():
                    for i, imgs in enumerate(val_loader):
                        # Configure model input
                        imgs_lr = Variable(imgs["lr"].type(Tensor))
                        imgs_hr = Variable(imgs["hr"].type(Tensor))
                        # Adversarial ground truths
                        if torch.cuda.device_count() > 1 and opt.parallel == 1:
                            valid = Variable(Tensor(np.ones((imgs_lr.size(0), *discriminator.module.output_shape))),
                                             requires_grad=False)
                            fake = Variable(Tensor(np.zeros((imgs_lr.size(0), *discriminator.module.output_shape))),
                                            requires_grad=False)
                        else:
                            valid = Variable(Tensor(np.ones((imgs_lr.size(0), *discriminator.output_shape))),
                                             requires_grad=False)
                            fake = Variable(Tensor(np.zeros((imgs_lr.size(0), *discriminator.output_shape))),
                                            requires_grad=False)

                        gen_hr = generator(imgs_lr)
                        loss_GAN = criterion_GAN(discriminator(gen_hr), valid)

                        # Content loss
                        if opt.channels == 1:  # Duplicate single channel to 3 channel tensor to match vgg19's expected input
                            fe_gen_hr = gen_hr.expand(-1, 3, -1, -1)
                            fe_imgs_hr = imgs_hr.expand(-1, 3, -1, -1)
                        else:
                            fe_gen_hr = gen_hr
                            fe_imgs_hr = imgs_hr

                        gen_features = feature_extractor(fe_gen_hr)
                        real_features = feature_extractor(fe_imgs_hr)

                        loss_content = temp_err(gen_hr, imgs_hr)

                        # Total loss
                        loss_G_v = loss_content + 1e-3 * loss_GAN

                        # Loss of real and fake images
                        loss_real = criterion_GAN(discriminator(imgs_hr), valid)
                        loss_fake = criterion_GAN(discriminator(gen_hr.detach()), fake)

                        # Total loss
                        loss_D_v = (loss_real + loss_fake) / 2

                        epoch_loss_g_val += loss_G_v.mean().item()
                        epoch_loss_d_val += loss_D_v.mean().item()

                    # Normalize losses and append
                    loss_g_val_norm = epoch_loss_g_val / len(val_loader)
                    loss_d_val_norm = epoch_loss_d_val / len(val_loader)
                    loss_g_val.append(loss_g_val_norm)
                    loss_d_val.append(loss_d_val_norm)

            if opt.checkpoint_interval != -1 and epoch % opt.checkpoint_interval == 0:
                # Save model checkpoints
                if torch.cuda.device_count() > 1 and opt.parallel == 1:
                    torch.save(generator.module.state_dict(), f"{saved_models}/generator_%d.pth" % epoch)
                    torch.save(discriminator.module.state_dict(), f"{saved_models}/discriminator_%d.pth" % epoch)
                else:
                    torch.save(generator.state_dict(), f"{saved_models}/generator_%d.pth" % epoch)
                    torch.save(discriminator.state_dict(), f"{saved_models}/discriminator_%d.pth" % epoch)

            # Update graph
            if opt.plot_loss == 1:
                x = list(np.arange(0, epoch + 1, 1))
                line1, = ax.plot(x, loss_g_train, 'r-', label='G training loss')
                line2, = ax.plot(x, loss_d_train, 'b-', label='D training loss')
                if opt.val_frac > 0.0:
                    line3, = ax.plot(x, loss_g_val, 'g-', label='G validation loss')
                    line4, = ax.plot(x, loss_d_val, 'y-', label='D validation loss')
                plt.legend(loc='upper left')
                fig.canvas.draw()
                fig.canvas.flush_events()
                ax.lines.remove(line1)
                ax.lines.remove(line2)
                if opt.val_frac > 0.0:
                    ax.lines.remove(line3)
                    ax.lines.remove(line4)

            # Append losses to log
            if opt.log_loss == 1:
                with open(f'{saved_models}/loss.txt', 'a') as file:
                    file.write(
                        f'{loss_g_train_norm};{loss_d_train_norm};{loss_g_val_norm};{loss_d_val_norm}\n') if opt.val_frac > 0.0 else file.write(
                        f'{loss_g_train_norm};{loss_d_train_norm}\n')


if __name__ == '__main__':
    main()