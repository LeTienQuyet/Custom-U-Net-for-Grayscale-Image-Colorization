# Image Colorization on CelebA-HQ Dataset
This project focuses on **Image Colorization** for human face images, primarily using a **U-Net** architecture, while also exploring the use of **PatchGAN** to potentially enhance the realism and overall visual quality of the generated colorized images.
## Experiments
|      Model         |  Loss Function |  PSNR $\uparrow$    |   SSIM $\uparrow$    |
| :---------------:  | :--------:     | :--------:  | :--------: |
| U-Net              | $\alpha \mathcal{L_2}$ + $\beta \mathcal{L_{percept}}$           |     **25.4007** |      **0.954**   |
| U-Net + PatchGAN   | $\alpha \mathcal{L_2}$ + $\beta \mathcal{L_{percept}}$ + $\theta \mathcal{L_{adv}} $ |      24.7444       |  0.9405 |
| U-Net + PatchGAN   | $\alpha \mathcal{L_1}$ + $\beta \mathcal{L_{percept}}$ + $\theta \mathcal{L_{adv}} $              |    24.9338         |    0.9454  |
| U-Net + PatchGAN   | $\alpha \mathcal{L_1}$  + $\theta \mathcal{L_{adv}} $              |   *updating*        |    *updating*  |