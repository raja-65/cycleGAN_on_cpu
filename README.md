----------
Here's a sample README.md file for your CycleGAN project:

**CycleGAN for Hubble Space Telescope Image Translation**
===========================================================

**Project Overview**
-------------------

This project uses CycleGAN to translate images from the Hubble Space Telescope's raw dataset to a processed dataset, and vice versa. The goal is to learn a mapping between the two domains, enabling the generation of high-quality images in both domains.

**Dataset**
------------

* Raw Hubble Space Telescope images (`.tif` files)
* Processed Hubble Space Telescope images (`.jpg` files)

**Model Architecture**
---------------------

* Generator: U-Net with 7 convolutional layers and 2 transposed convolutional layers
* Discriminator: PatchGAN with 4 convolutional layers and 1 sigmoid output layer

**Training**
------------

* Batch size: 1
* Number of epochs: 50
* Learning rate: 0.0002
* Optimizer: Adam with betas (0.5, 0.999)
* Loss functions: Mean Squared Error (MSE) for GAN loss, L1 loss for cycle loss and identity loss

**Results**
------------

* Translated images from raw to processed and vice versa
* Loss plots for generator and discriminator during training

**Usage**
---------

1. Clone the repository
2. Install required libraries: `torch`, `torchvision`, `numpy`, `matplotlib`
3. Download the raw and processed Hubble Space Telescope image datasets
4. Run `python cycle_gan.py` to train the model
5. Use the trained model to generate translated images

**Model Weights**
----------------

* Trained model weights are available on the Hugging Face Hub: [link to your HF model]
* Model weights can be downloaded and used for inference

**License**
------------

* This project is licensed under the  GNU GENERAL PUBLIC LICENSE

**Acknowledgments**
------------------

* This project was developed as part of the Bachelor's Thesis Project (BTP) at [Your University Name]
* Special thanks to [Your Supervisor's Name] for guidance and support

**References**
---------------

* [1] Zhu, J.-Y., Park, T., Isola, P., & Efros, A. A. (2017). Unpaired image-to-image translation using cycle-consistent adversarial networks. In Proceedings of the IEEE International Conference on Computer Vision (pp. 2223-2232).
* [2] Hubble Space Telescope dataset
