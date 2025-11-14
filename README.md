# Image Colorization on CelebA-HQ Dataset
This project focuses on **Image Colorization** for human face images, primarily using a **U-Net** architecture, while also exploring the use of **PatchGAN** to potentially enhance the realism and overall visual quality of the generated colorized images.
## Architecture
The model is based on a U-Net architecture (see the figure below) inspired by **UVCGAN** [1].
![unet_architecture](images/unet_architecture.png)
## Experiments
|$~~~~~~~$ Model $~~~~~~~$| $~~~~~~~~~~$ Loss Function $~~~~~~~$| PSNR ↑ | SSIM ↑ |
| :--: | :--: | :--: | :--: |
| U-Net | $\alpha \mathcal{L_1}$ + $\beta \mathcal{L_{percept}}$ | ... | ... |
| U-Net + PatchGAN | $\alpha \mathcal{L_1}$ + $\theta \mathcal{L_{adv}} $  | ... | ... |
| U-Net + PatchGAN | $\alpha \mathcal{L_1}$ + $\beta \mathcal{L_{percept}}$ + $\theta \mathcal{L_{adv}} $  | ... | ... |
| U-Net | $\alpha \mathcal{L_2}$ + $\beta \mathcal{L_{percept}}$ | ... | ... |
| U-Net + PatchGAN | $\alpha \mathcal{L_2}$ + $\theta \mathcal{L_{adv}} $ | ... | ... |
| U-Net + PatchGAN | $\alpha \mathcal{L_2}$ + $\beta \mathcal{L_{percept}}$ + $\theta \mathcal{L_{adv}} $ | ... | ... |
## References
[1] **Dmitrii Torbunov**, **Yi Huang**, **Haiwang Yu**, **Jin Huang**,
 **Shinjae Yoo**, **Meifeng Lin**, **Brett Viren**, **Yihui Ren** (2022). *UVCGAN: UNetVision Transformer cycle-consistent GAN for unpaired
 image-to-image translation*.  [[arXiv]](https://arxiv.org/pdf/2203.02557)
