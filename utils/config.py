import argparse

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("Boolean value expected.")

def get_args():
    parser = argparse.ArgumentParser(description="Hyper-parameters for training")

    parser.add_argument("--num_epochs", type=int, help="No. epochs for training", default=100)
    parser.add_argument("--ndims", type=int, help="No. dimension of UNet", default=16)
    parser.add_argument("--lr", type=float, help="Learning rate for optimizer", default=0.0002)
    parser.add_argument("--batch_size", type=int, help="Batch size of loading dataset", default=64)
    parser.add_argument("--save_pth", type=str, help="Folder save checkpoint", default="./output")
    parser.add_argument("--root", type=str, help="Data folder", default="./data")
    parser.add_argument("--img_size", type=int, help="Image size", default=128)
    parser.add_argument("--alpha", type=float, help="Weight of Pixel Loss", default=8.0)
    parser.add_argument("--beta", type=float, help="Weight of Perceptual Loss", default=1.0)
    parser.add_argument("--theta", type=float, help="Weight of Generator Loss", default=0.02)
    parser.add_argument("--color_mode", type=str, help="Use Lab or RGB for training", default="lab")
    parser.add_argument("--type_loss", type=str, help="L1 or MSE for pixel loss", default="L1")

    args = parser.parse_args()
    return args