# Image Colorization on CelebA-HQ Dataset
This project focuses on **Image colorization** for human face images, primarily using a **U-Net** architecture to generate realistic and visually appealing colorized results.
## Architecture
The model is based on a U-Net architecture (see the figure below) inspired by **UVCGAN** [1].
<p align="center">
    <img src="images/unet_architecture.png" alt="U-Net architecture"/>
</p>

## Experiments
The model is trained with the following settings: 
```python
# Training configuration
epoch = 200, batch_size = 64, ndims=16, num_blocks = 1
# Loss weights
alpha = 2.0, beta = 1.0, theta = 0.04
# Optimizer
optimizer = Adam(lr=0.0004, betas=(0.5, 0.999))
```
The following table presents a detailed comparison of the model's performance when trained using different combinations of loss functions:
<div align="center">

|  Loss Function | $\mathbf{PSNR}_{Lab}\uparrow$ | $\mathbf{SSIM}_{Lab}\uparrow$ |
| :--: | :--: | :--: |
|  |  |  |
|   | |  |
| |  |  |
|  |  |  |

</div>
The images below illustrate a visual comparison of the results obtained using different loss functions:
<p align="center">
    <img src="images/diff_loss.png" alt="Image for diff loss"/>
</p>

*Several experiments were conducted to evaluate different design choices in the colorization process. Two key observations emerged:*
* **Color space comparison**: Predicting **ab** channels from **L** yields better colorization results than directly predicting **RGB** values.
* **Effect of PatchGAN**: Incorporating **PatchGAN**—whether in the VanillaGAN or LSGAN setup—deteriorates the visual quality, leading to less realistic colorized images.
## References
[1] **Dmitrii Torbunov**, **Yi Huang**, **Haiwang Yu**, **Jin Huang**,
 **Shinjae Yoo**, **Meifeng Lin**, **Brett Viren**, **Yihui Ren** (2022). *UVCGAN: UNetVision Transformer cycle-consistent GAN for unpaired
 image-to-image translation*.  [[arXiv]](https://arxiv.org/pdf/2203.02557)
