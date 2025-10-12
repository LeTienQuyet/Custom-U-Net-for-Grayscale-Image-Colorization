import os
import argparse
import random

import torch
import matplotlib.pyplot as plt

from torchmetrics.image import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure
from tqdm import tqdm
from models.unet import UNet
from utils.dataset import prepare_data
from utils.loss import CombinedLoss

def get_args():
    parser = argparse.ArgumentParser(description="Hyper-parameters for training")

    parser.add_argument("--num_epochs", type=int, help="No. epochs for training", default=50)
    parser.add_argument("--ndims", type=int, help="No. dimension of UNet", default=8)
    parser.add_argument("--lr", type=float, help="Learning rate for optimizer", default=0.001)
    parser.add_argument("--alpha", type=float, help="Weight of average loss", default=0.01)
    parser.add_argument("--batch_size", type=int, help="Batch size of loading dataset", default=64)
    parser.add_argument("--save_pth", type=str, help="Folder save checkpoint", default="./output")
    parser.add_argument("--root", type=str, help="Data folder", default="./data")
    parser.add_argument("--img_size", type=int, help="Image size", default=256)

    args = parser.parse_args()
    return args

def plot_loss(num_epochs, train_losses, val_losses, save_pth):
    epochs = range(1, num_epochs+1)
    plt.figure(figsize=(10, 5))
    plt.plot(epochs, train_losses, label='Train Loss')
    plt.plot(epochs, val_losses, label='Validate Loss')
    plt.title('Training & Validate loss arcording to Epoch')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid()
    plt.savefig(os.path.join(save_pth, "loss.png"), dpi=300, bbox_inches='tight')

def val_model(model, val_dataloader, criterion, device):
    model.eval()

    total_val_mse_loss, total_val_percept_loss, total_val_loss = 0.0, 0.0, 0.0
    total_samples = len(val_dataloader.dataset)

    psnr_metric = PeakSignalNoiseRatio(data_range=1.0).to(device)
    ssim_metric = StructuralSimilarityIndexMeasure(data_range=1.0).to(device)

    total_psnr = 0.0
    total_ssim = 0.0

    with torch.no_grad():
        for inputs, targets in val_dataloader:
            inputs, targets = inputs.to(device), targets.to(device)
            batch = inputs.size(0)

            preds = model(inputs)
            val_mse_loss, val_percept_loss, val_loss = criterion(preds, targets)

            total_val_mse_loss += val_mse_loss.item() * batch
            total_val_percept_loss += val_percept_loss.item() * batch
            total_val_loss += val_loss.item() * batch

            preds_ = (preds + 1.0) / 2
            targets_ = (targets + 1.0) / 2

            total_psnr += psnr_metric(preds_, targets_).item() * batch
            total_ssim += ssim_metric(preds_, targets_).item() * batch

    avg_val_mse_loss = total_val_mse_loss / total_samples
    avg_val_percept_loss = total_val_percept_loss / total_samples
    avg_val_loss = total_val_loss / total_samples

    avg_val_psnr = total_psnr / total_samples
    avg_val_ssim = total_ssim / total_samples
    return avg_val_mse_loss, avg_val_percept_loss, avg_val_loss, avg_val_psnr, avg_val_ssim
def train_model(num_epochs, model, train_dataloader, val_dataloader, optimizer, scheduler, criterion, device, save_pth):
    os.makedirs(save_pth, exist_ok=True)

    train_losses, val_losses = [], []
    max_psnr = 0.0

    total_samples = len(train_dataloader.dataset)

    for epoch in range(1, num_epochs+1):
        model.train()
        total_train_mse_loss, total_train_percept_loss, total_train_loss = 0.0, 0.0, 0.0

        pbar = tqdm(train_dataloader, desc=f"Epoch {epoch}/{num_epochs}", unit="batch", colour="RED")

        for inputs, targets in pbar:
            inputs, targets = inputs.to(device), targets.to(device)
            batch = inputs.size(0)

            preds = model(inputs)
            train_mse_loss, train_percept_loss, train_loss = criterion(preds, targets)

            total_train_mse_loss += train_mse_loss.item() * batch
            total_train_percept_loss += train_percept_loss.item() * batch
            total_train_loss += train_loss.item() * batch

            optimizer.zero_grad()
            train_loss.backward()
            optimizer.step()

            pbar.set_postfix({"Total loss": f"{train_loss.item():.4f}"})

        avg_train_mse_loss = total_train_mse_loss / total_samples
        avg_train_percept_loss = total_train_percept_loss / total_samples
        avg_train_loss = total_train_loss / total_samples

        avg_val_mse_loss, avg_val_percept_loss, avg_val_loss, avg_val_psnr, avg_val_ssim = val_model(model, val_dataloader, criterion, device)

        train_losses.append(avg_train_loss)
        val_losses.append(avg_val_loss)

        # Step the scheduler
        scheduler.step()
        print(f"Train:    MSE loss = {avg_train_mse_loss:.4f}, Perceptual loss = {avg_train_percept_loss:.4f}, Total loss = {avg_train_loss:.4f}")
        print(f"Validate: MSE loss = {avg_val_mse_loss:.4f}, Perceptual loss = {avg_val_percept_loss:.4f}, Total loss = {avg_val_loss:.4f}")
        print(f"PSNR = {avg_val_psnr:.4f}, SSIM = {avg_val_ssim:.4f}")

        # Save last model
        torch.save(
            model.state_dict(),
            os.path.join(save_pth, "last_model.pt")
        )

        # Save best model
        if avg_val_psnr > max_psnr:
            max_psnr = avg_val_psnr
            torch.save(
                model.state_dict(),
                os.path.join(save_pth, "best_model.pt")
            )
            print(f"Save best model at Epoch {epoch} !!!\n")

    plot_loss(num_epochs, train_losses, val_losses, save_pth)

def main(num_epochs, ndims, lr, batch_size, alpha, save_pth, root, img_size):
    # Fixed random
    seed = 412
    random.seed(seed)
    torch.manual_seed(seed)

    # Preprare data for training
    train_dataloader = prepare_data(root=root, split="train", batch_size=batch_size, img_size=img_size)
    val_dataloader = prepare_data(root=root, split="dev", batch_size=batch_size, img_size=img_size)
    print(f"[DATASET] {len(train_dataloader.dataset)} train samples & {len(val_dataloader.dataset)} validate samples")

    # Prepare model, optimizer and loss function
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = UNet(ndims=ndims).to(device)
    print(f"[UNET]    {sum(p.numel() for p in model.parameters() if p.requires_grad)} trainable params")

    optimizer = torch.optim.Adam(model.parameters(), lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)

    criterion = CombinedLoss(alpha=alpha).to(device)

    # Training phase
    train_model(
        num_epochs, model, train_dataloader, val_dataloader,
        optimizer, scheduler, criterion, device, save_pth
    )

if __name__ == "__main__":
    args = get_args()

    main(
        num_epochs=args.num_epochs, lr=args.lr, batch_size=args.batch_size,
        alpha=args.alpha, ndims=args.ndims, save_pth=args.save_pth, root=args.root,
        img_size=args.img_size
    )