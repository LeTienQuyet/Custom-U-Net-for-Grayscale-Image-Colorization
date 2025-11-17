# Image Colorization on CelebA-HQ Dataset
This project focuses on **Image Colorization** for human face images, primarily using a **U-Net** architecture, while also exploring the use of **PatchGAN** to potentially enhance the realism and overall visual quality of the generated colorized images.
## Architecture
The model is based on a U-Net architecture (see the figure below) inspired by **UVCGAN** [1].
![unet_architecture](images/unet_architecture.png)
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

| Model |  Loss Function | PSNR ↑ | SSIM ↑ |
| :--: | :--: | :--: | :--: |
| U-Net | $\alpha \mathcal{L_1}$ + $\beta \mathcal{L_{percept}}$ | 24.7722 | 0.9438 |
| U-Net + PatchGAN | $\alpha \mathcal{L_1}$ + $\beta \mathcal{L_{percept}}$ + $\theta \mathcal{L_{adv}} $  | **25.2202**| **0.9569** |
| U-Net | $\alpha \mathcal{L_2}$ + $\beta \mathcal{L_{percept}}$ | 24.5076 | 0.9397 |
| U-Net + PatchGAN | $\alpha \mathcal{L_2}$ + $\beta \mathcal{L_{percept}}$ + $\theta \mathcal{L_{adv}} $ | 25.13 | 0.95 |

</div>

The images below illustrate a visual comparison of the results obtained using different loss functions:
![diff_loss](images/diff_loss.png)
## References
[1] **Dmitrii Torbunov**, **Yi Huang**, **Haiwang Yu**, **Jin Huang**,
 **Shinjae Yoo**, **Meifeng Lin**, **Brett Viren**, **Yihui Ren** (2022). *UVCGAN: UNetVision Transformer cycle-consistent GAN for unpaired
 image-to-image translation*.  [[arXiv]](https://arxiv.org/pdf/2203.02557)
