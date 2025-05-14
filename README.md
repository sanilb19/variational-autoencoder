# Variational Autoencoders (VAE) Demonstration

This repository contains two scripts that demonstrate the implementation and functionality of Variational Autoencoders (VAEs) using different frameworks and datasets. The purpose is to showcase a basic understanding of VAEs and their application in data generation.

## Repository Structure

```
generate2dImages/
│
├── vae_animation-1/
│   ├── vae_amination-1.py
│   └── outputs/
│       ├── latent_space_epoch_{epoch}.png
│       ├── final_latent_space.png
│       └── generated_digits.png
│
├── animated-gif/
│   ├── animated-gif.py
│   └── outputs/
│       └── vae_training.gif
│
└── README.md
```

## Scripts Overview

### 1. `vae_animation-1.py`

- **Framework**: TensorFlow
- **Dataset**: MNIST
- **Description**: This script implements a VAE to learn the latent space representation of the MNIST dataset. It visualizes the latent space evolution and generates new digit images from the learned latent space.
- **Outputs**: 
  - Latent space visualizations saved as PNG files.
  - Generated digit images saved as PNG files.

### 2. `animated-gif.py`

- **Framework**: PyTorch
- **Dataset**: 2D Moons
- **Description**: This script implements a VAE to learn the latent space representation of a 2D dataset generated using the `make_moons` function. It creates an animated GIF showing the training process and the evolution of the latent space.
- **Outputs**: 
  - An animated GIF of the training process saved in the `outputs` directory.

## Installation

To run these scripts, you need to have Python installed along with the following packages:

- TensorFlow
- PyTorch
- Matplotlib
- Scikit-learn

You can install the required packages using pip:

```bash
pip install tensorflow torch matplotlib scikit-learn
```

## Usage

### Running `vae_animation-1.py`

Navigate to the `vae_animation-1` directory and run the script:

```bash
cd vae_animation-1
python vae_amination-1.py
```

The script will generate and save visualizations of the latent space and generated digits in the `outputs` directory.

### Running `animated-gif.py`

Navigate to the `animated-gif` directory and run the script:

```bash
cd animated-gif
python animated-gif.py
```

The script will create and save an animated GIF of the training process in the `outputs` directory.

## Results

- **TensorFlow VAE**: Visualizes the latent space of the MNIST dataset and generates new digit images.
- **PyTorch VAE**: Animates the training process of a VAE on a 2D dataset, showing the evolution of the latent space.

## Conclusion

This repository demonstrates a basic understanding of VAEs and their application in data generation using both TensorFlow and PyTorch. The scripts provide a foundation for further exploration and experimentation with VAEs. 