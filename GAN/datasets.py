import os
import cv2
import glob
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as transforms

"""
Creating (paired) datasets for GAN
"""

# For MSE loss function on single channel images
mean = 0.5
std = 0.5


class ImageDataset(Dataset):
    def __init__(self, root, hr_shape, factor):
        """
        Handles single-channel images (grayscale), applying different transformations depending on the image format.
        :param root: string
            Directory containing the images
        :param hr_shape: tuple
            Tuple that specifies the X, Y shape of the HR images
        :param factor: int
            Factor to downsample HR images with
        """
        hr_height, hr_width = hr_shape
        # Transforms for lr and hr images
        self.lr_transform = transforms.Compose(
            [
                transforms.Resize((hr_height // factor, hr_height // factor)),
                transforms.ToTensor(),
                transforms.Normalize(mean, std),
            ]
        )
        self.hr_transform = transforms.Compose(
            [
                transforms.Resize((hr_height, hr_height)),
                transforms.ToTensor(),
                transforms.Normalize(mean, std),
            ]
        )
        self.cv2_transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(mean, std)
            ]
        )
        self.files = sorted(glob.glob(root + "/*.*"))
        self.factor = factor
        self.hr_height, self.hr_width = hr_shape

    def __getitem__(self, index):
        """
        Loads and processes an image at the given index.
        :param index: int
        :return: dictionary
            dictionary with downsampled HR and HR images
        """
        img = Image.open(self.files[index % len(self.files)])
        if img.mode == "L":
            img_lr = self.lr_transform(img)
            img_hr = self.hr_transform(img)
        else:
            img = cv2.imread(self.files[index % len(self.files)], cv2.IMREAD_GRAYSCALE)
            img_lr = self.resize_cv2(img, self.hr_height, self.hr_width, self.factor)
            img_lr = self.cv2_transform(img_lr)
            img_hr = self.cv2_transform(img)

        return {"lr": img_lr, "hr": img_hr}

    def __len__(self):
        return len(self.files)

    def resize_cv2(self, img, hr_height, hr_width, factor):
        resized_img = cv2.resize(img, (hr_height // factor, hr_height // factor), interpolation=cv2.INTER_CUBIC)
        return resized_img


class PairedImageDataset(Dataset):
    """
    Handles paired high-resolution and low-resolution image datasets.
    """
    def __init__(self, hr_img_dir, lr_img_dir):
        """
        Converts images to tensor and normalizes them
        :param hr_img_dir: string
            HR images path
        :param lr_img_dir: string
            LR images path
        """
        self.hr_img_dir = hr_img_dir
        self.lr_img_dir = lr_img_dir
        self.all_hr_files = os.listdir(self.hr_img_dir)
        self.all_lr_files = os.listdir(self.lr_img_dir)
        self.transform = transforms.Compose([
                        transforms.ToTensor(),
                        transforms.Normalize(mean, std)
                        ])

    def __len__(self):
        return len(self.all_hr_files)

    def __getitem__(self, idx):
        """
        Loads and processes high-resolution and low-resolution images at the given index.
        :param idx: index
        :return: dictionary
            Dictionary with paired LR and HR image
        """
        selected_hr_file = self.all_hr_files[idx]
        selected_lr_file = self.all_lr_files[idx]
        hr_img = Image.open(os.path.join(self.hr_img_dir, selected_hr_file), mode='r')
        lr_img = Image.open(os.path.join(self.lr_img_dir, selected_lr_file), mode='r')

        if self.transform is not None:
            hr_img = self.transform(hr_img)
            lr_img = self.transform(lr_img)
        sample = {'lr': lr_img,
                  'hr': hr_img}
        return sample
