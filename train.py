import os
import random

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from torchmetrics.image.fid import FrechetInceptionDistance
from torchmetrics.image import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure
from tqdm import tqdm

from models import UNet, PatchGAN
from utils import (
    PerceptualLoss,
    ColorfulnessMetric,
    epoch_summary,
    prepare_data,
    lab2rgb,
    get_args
)

def val_model(args):
    args.generator.eval()

    total_samples = len(args.val_dataloader.dataset)

    psnr_metric = PeakSignalNoiseRatio(data_range=1.0).to(args.device)
    ssim_metric = StructuralSimilarityIndexMeasure(data_range=1.0).to(args.device)
    fid_metric = FrechetInceptionDistance(feature=2048, normalize=True).to(args.device)

    total_psnr = total_ssim = total_delta_cf = 0.0

    with torch.no_grad():
        for inputs, targets in args.val_dataloader:
            inputs, targets = inputs.to(args.device), targets.to(args.device)
            batch = inputs.size(0)

            preds = args.generator(inputs)

            if args.color_mode == "lab":
                preds_for_metrics = lab2rgb(inputs, preds)
                targets_for_metrics = lab2rgb(inputs, targets)
            else:
                preds_for_metrics = preds
                targets_for_metrics = targets

            total_psnr += psnr_metric(preds_for_metrics, targets_for_metrics).item() * batch
            total_ssim += ssim_metric(preds_for_metrics, targets_for_metrics).item() * batch

            cf_preds = ColorfulnessMetric(preds_for_metrics, reduction="sum")
            cf_targets = ColorfulnessMetric(targets_for_metrics, reduction="sum")
            total_delta_cf += torch.abs(cf_preds-cf_targets).item()

            # FID Score
            preds_for_metrics = F.interpolate(preds_for_metrics, size=(299, 299), mode="bilinear", align_corners=False)
            targets_for_metrics = F.interpolate(targets_for_metrics, size=(299, 299), mode="bilinear", align_corners=False)

            fid_metric.update(targets_for_metrics, real=True)
            fid_metric.update(preds_for_metrics, real=False)

    avg_val_fid = fid_metric.compute().item()
    avg_val_psnr = total_psnr / total_samples
    avg_val_ssim = total_ssim / total_samples
    avg_val_delta_cf = total_delta_cf / total_samples
    return avg_val_fid, avg_val_psnr, avg_val_ssim, avg_val_delta_cf

def train_model(args):
    os.makedirs(args.save_pth, exist_ok=True)

    min_fid = float("inf")
    psnr = ssim = delta_cf = 0.0

    total_samples = len(args.train_dataloader.dataset)

    # Determine training mode
    use_gan = args.theta != 0
    use_perceptual = args.beta != 0

    for epoch in range(1, args.num_epochs + 1):
        args.generator.train()
        if use_gan:
            args.discriminator.train()

        total_train_pixel_loss = 0.0
        total_train_percept_loss = 0.0 if use_perceptual else None
        total_train_gen_loss = 0.0 if use_gan else None

        pbar = tqdm(args.train_dataloader, desc=f"Epoch {epoch}/{args.num_epochs}", unit="batch", colour="GREEN", ncols=150)

        for inputs, targets in pbar:
            inputs, targets = inputs.to(args.device), targets.to(args.device)
            batch = inputs.size(0)

            # Generate predictions
            preds = args.generator(inputs)

            # Training Discriminator (only if using GAN)
            if use_gan:
                args.disOptimizer.zero_grad()

                preds_concat = torch.cat([inputs, preds], dim=1)
                targets_concat = torch.cat([inputs, targets], dim=1)

                # For real data
                real_outputs = args.discriminator(targets_concat)
                real_labels = torch.ones_like(real_outputs)
                train_lossDis_real = args.gen_criterion(real_outputs, real_labels)

                # For fake data
                fake_outputs = args.discriminator(preds_concat.detach())  # Prevent gradient to Generator
                fake_labels = torch.zeros_like(fake_outputs)
                train_lossDis_fake = args.gen_criterion(fake_outputs, fake_labels)

                train_lossDis = 0.5 * (train_lossDis_real + train_lossDis_fake)
                train_lossDis.backward()
                args.disOptimizer.step()

            # Training Generator
            args.genOptimizer.zero_grad()

            # Compute losses
            train_pixel_loss = args.pixel_criterion(preds, targets)

            # Perceptual loss (if enabled)
            if use_perceptual:
                if args.color_mode == "lab":
                    preds_for_perceptual = lab2rgb(inputs, preds)
                    targets_for_perceptual = lab2rgb(inputs, targets)
                else:
                    preds_for_perceptual = preds
                    targets_for_perceptual = targets

                train_percept_loss = args.perceptual_criterion(preds_for_perceptual, targets_for_perceptual)
                total_train_percept_loss += train_percept_loss.detach().item() * batch

            # GAN loss (if enabled)
            if use_gan:
                preds_concat = torch.cat([inputs, preds], dim=1)
                fake_outputs = args.discriminator(preds_concat)
                fake_labels = torch.ones_like(fake_outputs)
                train_gen_loss = 0.5 * args.gen_criterion(fake_outputs, fake_labels)
                total_train_gen_loss += train_gen_loss.detach().item() * batch

            # Combine losses
            train_loss = args.alpha * train_pixel_loss
            if use_perceptual:
                train_loss += args.beta * train_percept_loss
            if use_gan:
                train_loss += args.theta * train_gen_loss

            train_loss.backward()
            args.genOptimizer.step()

            total_train_pixel_loss += train_pixel_loss.detach().item() * batch

            # Update progress bar
            postfix_dict = {'Pixel': f'{train_pixel_loss.item():.4f}'}
            if use_perceptual:
                postfix_dict['Perceptual'] = f'{train_percept_loss.item():.4f}'
            if use_gan:
                postfix_dict['Gen'] = f'{train_gen_loss.item():.4f}'
            pbar.set_postfix(postfix_dict)

        # Calculate average losses
        avg_train_pixel_loss = total_train_pixel_loss / total_samples
        avg_train_percept_loss = total_train_percept_loss / total_samples if use_perceptual else None
        avg_train_gen_loss = total_train_gen_loss / total_samples if use_gan else None

        avg_val_fid, avg_val_psnr, avg_val_ssim, avg_val_delta_cf = val_model(args)

        # Step the Scheduler
        args.genScheduler.step()
        if use_gan:
            args.disScheduler.step()

        # Information
        epoch_summary(
            epoch=epoch, avg_val_fid=avg_val_fid, avg_val_psnr=avg_val_psnr, avg_val_ssim=avg_val_ssim, avg_val_delta_cf=avg_val_delta_cf,
            train_mse_loss=avg_train_pixel_loss, train_percept_loss=avg_train_percept_loss, train_gen_loss=avg_train_gen_loss
        )

        # Save last model
        torch.save(
            args.generator.state_dict(),
            os.path.join(args.save_pth, f"generator_last.pt")
        )

        # Save best model
        if avg_val_fid < min_fid:
            min_fid, psnr, ssim, delta_cf = avg_val_fid, avg_val_psnr, avg_val_ssim, avg_val_delta_cf
            torch.save(
                args.generator.state_dict(),
                os.path.join(args.save_pth, f"generator_best.pt")
            )

            print(f"\nSave best model at Epoch {epoch} !!!\n")

    print(f"\nCompleted training with best FID = {min_fid:.4f}, PSNR = {psnr:.4f}, SSIM = {ssim:.4f}, ΔCF = {delta_cf:.4f}!!!")

def main(args):
    # Fixed random
    seed = 412
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)

    # Check color mode for training
    args.color_mode = args.color_mode.lower()
    if args.color_mode not in ("rgb", "lab"):
        raise ValueError(f"Invalid color_mode '{args.color_mode}' => Expected 'rgb' or 'lab' !")

    print(f"[COLOR] Training {args.color_mode} image with {args.type_loss} loss !")

    # Preprare data for training
    args.train_dataloader = prepare_data(root=args.root, split="train", batch_size=args.batch_size, img_size=args.img_size, color_mode=args.color_mode)
    args.val_dataloader = prepare_data(root=args.root, split="dev", batch_size=args.batch_size, img_size=args.img_size, color_mode=args.color_mode)
    print(f"[DATASET] {len(args.train_dataloader.dataset):,} train samples & {len(args.val_dataloader.dataset):,} validate samples !")

    args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load model
    if args.color_mode == "lab":
        args.generator = UNet(ndims=args.ndims, out_channels=2, depths=args.depths, num_blocks=args.num_blocks).to(args.device)
    else:
        args.generator = UNet(ndims=args.ndims, out_channels=3, depths=args.depths, num_blocks=args.num_blocks).to(args.device)

    print(f"[GENERATOR] {sum(p.numel() for p in args.generator.parameters() if p.requires_grad):,} trainable params !")

    # Optimizer and Scheduler
    args.genOptimizer = optim.Adam(args.generator.parameters(), args.lr, (0.5, 0.999))
    args.genScheduler = torch.optim.lr_scheduler.CosineAnnealingLR(args.genOptimizer, T_max=args.num_epochs)

    # Loss function
    args.type_loss = args.type_loss.upper()
    if args.type_loss == "L1":
        args.pixel_criterion = nn.L1Loss().to(args.device)
    elif args.type_loss in ("L2", "MSE"):
        args.pixel_criterion = nn.MSELoss().to(args.device)
    else:
        raise ValueError(f"Invalid type_loss '{args.type_loss}' => Expected 'L1', 'L2', or 'MSE' !")

    # Training phase
    if args.theta != 0:
        # Train with GAN
        args.gen_criterion = nn.MSELoss().to(args.device)

        if args.color_mode == "lab":
            args.discriminator = PatchGAN(in_channels=3).to(args.device)
        else:
            args.discriminator = PatchGAN(in_channels=4).to(args.device)

        args.disOptimizer = optim.Adam(args.discriminator.parameters(), args.lr / 4, (0.5, 0.999))
        args.disScheduler = torch.optim.lr_scheduler.CosineAnnealingLR(args.disOptimizer, T_max=args.num_epochs)

        print(f"[DISCRIMINATOR] {sum(p.numel() for p in args.discriminator.parameters() if p.requires_grad):,} trainable params !")

        if args.beta != 0:
            print("[TRAINING] U-Net with Perceptual and GAN !!!\n")
            args.perceptual_criterion = PerceptualLoss().to(args.device)
        else:
            print("[TRAINING] U-Net with GAN !!!\n")
    else:
        # Train without GAN
        print("[TRAINING] U-Net with Perceptual !!!\n")
        args.perceptual_criterion = PerceptualLoss().to(args.device)

    # Training
    train_model(args)

if __name__ == "__main__":
    args = get_args()

    main(args)