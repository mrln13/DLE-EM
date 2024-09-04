# DLE-EM: Deep-Learning-Enhanced Electron Microscopy

This is the official repository for the paper Deep-Learning-Enhanced Electron Microscopy for Earth Material Characterization.

## Introduction

The research introduces a Deep Learning (DL) based method to accelerate imaging of rock microstructures by up to 16 times with minimal quality loss.

**Abstract**

Rocks, as Earth materials, contain intricate microstructures that reveal their geological history.
These microstructures include grain boundaries, preferred orientation, twinning and porosity, 
and they hold critical significance in the realm of the energy transition. As they influence the physical strength, 
chemical reactivity, and fluid flow properties of rocks, they also directly impact subsurface reservoirs used for geothermal energy, 
nuclear waste disposal, and hydrogen and carbon dioxide storage. Understanding microstructures, 
and their distribution, is therefore essential for ensuring the stability and effectiveness of these subsurface activities. 
Achieving statistical representativeness often requires the imaging of a substantial quantity of samples at a high level of
magnification. To address this challenge, this research introduces a novel image enhancement process for scanning
electron microscopy datasets, demonstrating the potential for significant resolution improvement through
Deep-Learning-Enhanced Electron Microscopy (DLE-EM). This approach accelerates imaging processes,
up to a factor of 16, with minimal impact on quality and offers possibilities for real-time super-resolution imaging of
unknown microstructures, promising to advance the field of geoscience and material science.

**Workflow Overview**

![DLE-EM workflow overview](/workflow.png)

Overview of the DLE-EM workflow applied to the gabbro dataset. Panels **a**, **b**, and **c** schematically illustrate the process of generating training data, 
having a HR area within a LR dataset. Panel **d** showcases how the training data is utilized to construct both the *discriminator* and
*generator* models. Panel **e** demonstrates the application of the *generator model*, transforming LR input data into HR output data.


## Installation and Get Started

The workflow consists of two distinct parts:
1. The Image Registration procedure
2. The Deep Learning workflow


Currently, these components need to be executed in separate Python environments. You can create these environments using 
Anaconda Navigator by importing the *DLE-EM_gan.yaml* and *DLE-EM_imreg.yaml* files, respectively. After downloading the 
repository, navigate to the *Image_Registration* folder and run *ImageReg.py* for the Image Registration procedure. For 
the Deep Learning workflow, go to the GAN folder and run *srgan.py* to train a model, or *inference.py* to apply an already 
trained model to a dataset.

Alternatively, download and set up the environments by following the following steps:

```shell
git clone https://github.com/mrln13/DLE-EM.git 
conda env create -f DLE-EM_imreg.yaml
conda env create -f DLE-EM_gan.yaml
```

Next, activate either the Image Registration environment
```shell
conda activate DLE-EM_imreg
```
or the Deep Learning environment
```shell
conda activate DLE-EM_gan
```


In the near future, the package will include an installer that sets up a single Python environment for the entire workflow,
streamlining the setup process.


## Data Preparation

This package works with TIF(F) and CZI (Carl Zeiss Image) files, provided that: 
- The low resolution (LR) file completely contains the high resolution (HR) file (or files)
- The resolution disparity between HR and LR is (approximately) 2, 4, or 8

## Image Registration

Image registration can be performed on one or multiple HR maps that are located within a larger LR map. Make sure the
appropriate Python environment is activated, and then run *ImageReg.py*:
```shell
conda activate DLE-EM_imreg
cd Image_Registration
python ImageReg.py
```

Follow the Prompts:

a. Select Intended Use:

* Choose from the following options:
  1. Locate HR tile(s) in LR map and extract files.
  2. Image registration of processed HR and LR tiles.
  3. Do both.
  4. Cancel.

b. Locate HR Tiles in LR Map (if applicable):
- Select the LR file.
- Indicate if there are multiple HR regions in the LR map.
- Select the HR file(s).

c. Process Image Pairs (if applicable):
- Specify the number of paired files to process.
- Select the corresponding LR and HR files.

d. Specify Resolution Disparity Factor:
- Provide the resolution disparity factor (2, 4, or 8).

The script performs image registration and correction using the specified or default parameters. 
Upon completion, the script saves the results in a specified folder.

Additional arguments can be provided. For more details, run:
```shell
python ImageReg.py --help
```
The intermediate and final results of the image registration procedure are saved in a folder with a timestamp appended to its name
and can be used to generate training data.

## Generating Training Data

Activate the appropriate Python environment and navigate to the GAN folder:
```shell
conda activate DLE-EM_gan
cd GAN
```

The Deep Learning process relies on training with paired HR and LR data, where the HR tiles are 256x256 pixels. The training data
is generated by a preprocessing script.

```shell
python preprocess.py
```

Follow the Prompts:

a. Select Image Source:
* Indicate if you are using ImageReg processed files or other registered image pairs.
* If using ImageReg, select the ImageReg.txt file when prompted.
* If using other sources, specify the number of paired maps and select each LR and HR image file accordingly.

b. Set Aside Testing Data:
- Specify the percentage of data to reserve for testing. This data will not be used for training or validation. Using the
testing data, model performance can be assessed.

c. Data Augmentation:
- Choose whether to augment the data (only necessary for small datasets). 
- If augmenting, decide whether to use default parameters or specify custom parameters (number of angles, number of lines, samples per line, and tile padding fraction).

d. Select Output Folder:
- Choose the folder where the preprocessing output will be saved.

The script will process the specified image pairs and save the results in the selected output folder.
Upon completion, the script will display the count of training and testing image pairs created.

## Model Training

When the script is called, it requires a folder containing the paired HR/LR training dataset. 
Model training is initiated with default parameters, which can be customized via command-line arguments.

Make sure the correct Python environment is active, and you are in the GAN folder before running *srgan.py*. Using the default parameters:
```shell
python srgan.py
```
Or with example custom parameters:

```shell
python srgan.py --dataset_name MyDataset --n_epochs 50 --batch_size 64 --lr 0.0002 --factor 2

```
For details on the available parameters, run:
```shell
python srgan.py --help
```

When running the script, the user will be prompted to provide the folder that contains the paired HR and LR training data.
The following output will be generated:
* Model Checkpoints: Saved in a datetime-stamped folder (e.g., *2024_08_05_10_30_saved_models/*).
* Generated Images: Saved in a datetime-stamped folder (e.g., *2024_08_05_10_30_images/*).
* Loss Logs: Saved in *loss.txt* within the model checkpoint folder.


Resume training from a specified epoch by providing the --epoch=n argument.

```shell
python srgan.py --epoch 10 --dataset_name MyDataset --n_epochs 50

```


## Inference

This script is designed for upscaling a single LR image (e.g., a map) using a pretrained generative model. Alternatively, the script can be used to process
paired HR and LR tiles to evaluate model performance.

```shell
python inference.py
```

The user is then prompted to select a pretrained model, the resolution disparity between HR and LR, and the processing mode (single map, or paired dataset).
The processed images are saved in the specified directory. For maps, output can be either as upscaled tiles, a complete upscaled map, or both.
For paired datasets, the output includes the generated images and, optionally, the original LR/HR images and composite grids.


Several command line arguments are available. For more information, run:
```shell
python inference.py --help
```

Example usage:
```shell
python inference.py --tile_size 200 --seamless False
```
