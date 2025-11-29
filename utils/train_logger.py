def epoch_summary(epoch, avg_val_fid, avg_val_psnr, avg_val_ssim, avg_val_delta_cf, train_mse_loss, train_percept_loss=None, train_gen_loss=None, width=75):
    line = "-" * width
    # Epoch
    print(line)
    epoch_str = f"Epoch = {epoch}"
    print(f"|{epoch_str:^{width - 2}}|")
    print(line)

    # Train loss
    loss_str = f"Loss → Pixel: {train_mse_loss:.4f}"

    if train_percept_loss is not None:
        loss_str += f"    Perceptual: {train_percept_loss:.4f}"

    if train_gen_loss is not None:
        loss_str += f"    Gen: {train_gen_loss:.4f}"
    print(f"|{loss_str:^{width - 2}}|")
    print(line)

    # Validation metrics
    metric_str = f"Metrics → FID = {avg_val_fid:.4f}   PSNR: {avg_val_psnr:.4f}   SSIM: {avg_val_ssim:.4f}   ΔCF: {avg_val_delta_cf:.4f}"
    print(f"|{metric_str:^{width - 2}}|")
    print(line)