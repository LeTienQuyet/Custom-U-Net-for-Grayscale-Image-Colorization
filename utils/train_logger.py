# def log_table(epoch, train_mse_loss, train_percept_loss, train_gen_loss, avg_val_psnr, avg_val_ssim, width=53):
#     # Epoch
#     print("-" * width)
#     epoch_str = f"Epoch = {epoch}"
#     print(f"|{epoch_str:^{width - 2}}|")
#     print("-" * width)
#
#     # Train loss
#     loss_str = f"Pixel: {train_mse_loss:.4f}    Perceptual: {train_percept_loss:.4f}   Gen: {train_gen_loss:.4f}"
#     print(f"|{loss_str:^{width - 2}}|")
#     print("-" * width)
#
#     # Validation metrics
#     metric_str = f"PSNR: {avg_val_psnr:.4f}   SSIM: {avg_val_ssim:.4f}"
#     print(f"|{metric_str:^{width - 2}}|")
#     print("-" * width)

def epoch_summary(epoch, train_mse_loss, train_percept_loss, avg_val_psnr, avg_val_ssim, width=53):
    # Epoch
    print("-" * width)
    epoch_str = f"Epoch = {epoch}"
    print(f"|{epoch_str:^{width - 2}}|")
    print("-" * width)

    # Train loss
    loss_str = f"Pixel: {train_mse_loss:.4f}    Perceptual: {train_percept_loss:.4f}"
    print(f"|{loss_str:^{width - 2}}|")
    print("-" * width)

    # Validation metrics
    metric_str = f"PSNR: {avg_val_psnr:.4f}   SSIM: {avg_val_ssim:.4f}"
    print(f"|{metric_str:^{width - 2}}|")
    print("-" * width)