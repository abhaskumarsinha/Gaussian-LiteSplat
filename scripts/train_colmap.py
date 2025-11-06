#!/usr/bin/env python3
import os
import sys
import argparse
import logging
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from tqdm.auto import tqdm

# Add project root to sys.path for relative imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# Force torch backend
os.environ["KERAS_BACKEND"] = "torch"

from litesplat.io import convert_colmap_to_gaussians
from litesplat.core import GaussianParameterLayer, CameraLayer
from litesplat.utils import Renderer


def setup_logger(log_level=logging.INFO):
    """Setup a colored console logger."""
    logger = logging.getLogger("LiteSplatTrain")
    handler = logging.StreamHandler(sys.stdout)
    formatter = logging.Formatter("[%(asctime)s] [%(levelname)s] %(message)s", "%H:%M:%S")
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(log_level)
    return logger


def parse_args():
    parser = argparse.ArgumentParser(description="Train Gaussian-LiteSplat on COLMAP dataset")
    parser.add_argument("--colmap_scene", type=str, required=True,
                        help="Path to COLMAP scene directory")
    parser.add_argument("--litesplat_scene", type=str, required=True,
                        help="Name of converted LiteSplat scene")
    parser.add_argument("--images_dir", type=str, required=True,
                        help="Path to image directory")

    parser.add_argument("--total_gaussians", type=int, default=2560)
    parser.add_argument("--max_gaussians", type=int, default=2560)

    parser.add_argument("--trainable_focus", action="store_true")
    parser.add_argument("--trainable_principal", action="store_true")
    parser.add_argument("--trainable_extrinsics", action="store_true")

    parser.add_argument("--output_h", type=int, default=int(48 * 1.5))
    parser.add_argument("--output_w", type=int, default=int(64 * 1.5))
    parser.add_argument("--output_gaussians", type=str, default="gaussians.JSON")

    parser.add_argument("--cam_import", type=int, default=45)
    parser.add_argument("--train_cams", type=int, default=15)
    parser.add_argument("--lr", type=float, default=1e-2)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--num_epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--checkpoint", type=int, default=10)

    parser.add_argument("--num_images", type=int, default=15)
    parser.add_argument("--cols", type=int, default=5)
    parser.add_argument("--output_dir", type=str, default="output")
    parser.add_argument("--dpi", type=int, default=300)
    parser.add_argument("--torch_compile", action="store_true")

    parser.add_argument("--log_level", type=str, default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])

    return parser.parse_args()


def main():
    args = parse_args()
    logger = setup_logger(getattr(logging, args.log_level.upper()))

    # --- Step 1: Convert COLMAP scene ---
    logger.info(f"Converting COLMAP scene: {args.colmap_scene}")
    convert_colmap_to_gaussians(args.colmap_scene, args.litesplat_scene)

    # --- Step 2: Create Gaussian layer ---
    logger.info("Initializing Gaussian parameters...")
    gaussians = GaussianParameterLayer(args.litesplat_scene, total_gaussians=args.total_gaussians)

    # --- Step 3: Create camera layers ---
    logger.info("Loading camera layers...")
    cams = []
    for i in range(args.cam_import):
        cams.append(
            CameraLayer(
                args.litesplat_scene,
                gaussians,
                args.images_dir,
                args.trainable_focus,
                args.trainable_principal,
                args.trainable_extrinsics,
                i,
                args.max_gaussians,
                args.output_h,
                args.output_w,
            )
        )

    # --- Step 4: Renderer setup ---
    renderer = Renderer(cams, args.output_h, args.output_w, args.torch_compile)

    # --- Step 5: Select training cameras ---
    cam_ids = [int(i) for i in range(args.cam_import)]
    random.shuffle(cam_ids)
    cam_ids = cam_ids[:args.train_cams]
    logger.info(f"Selected {len(cam_ids)} cameras for training: {cam_ids}")

    # --- Step 6: Prepare target images ---
    y_real = torch.stack([cams[i].y_real for i in cam_ids], dim=0) / 255.0
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    y_real = y_real.to(device)
    renderer = renderer.to(device)
    logger.info(f"y_real shape: {tuple(y_real.shape)}  Device: {device}")

    # --- Step 7: Setup optimizer and loss ---
    optimizer = optim.Adam(renderer.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    criterion = nn.MSELoss()

    # --- Step 8: Training loop ---
    logger.info("Starting training loop...")
    cam_ids_tensor = torch.arange(args.train_cams, device=device)

    for epoch in tqdm(range(args.num_epochs), desc="Training"):
        renderer.train()
        optimizer.zero_grad()

        y_pred = renderer.call(cam_ids_tensor, args.output_h, args.output_w, render=False, bsz=args.batch_size)
        y_pred = y_pred.to(device).float()

        loss = criterion(y_pred, y_real)
        loss.backward(retain_graph=True)
        optimizer.step()

        if (epoch + 1) % args.checkpoint == 0 or epoch == 0:
            logger.info(f"Epoch [{epoch + 1}/{args.num_epochs}]  Loss: {loss.item():.6f}")
            for cam in cams:
                cam.get_sorted_keys()
            base, ext = os.path.splitext(args.output_gaussians)
            epoch_filename = f"{base}_epoch{epoch}{ext}"
            cams[0].save_gaussians(epoch_filename)

        del y_pred, loss
        torch.cuda.empty_cache()

    # --- Step 9: Render and save images ---
    logger.info("Rendering final images...")
    with torch.no_grad():
        imgs = renderer.call(cam_ids_tensor, args.output_h, args.output_w, render=False, bsz=args.batch_size)

    rows = int(np.ceil(args.num_images / args.cols))
    os.makedirs(args.output_dir, exist_ok=True)
    output_path = os.path.join(args.output_dir, "image_grid.png")

    fig, axes = plt.subplots(rows, args.cols, figsize=(args.cols * 3, rows * 3))
    for i, ax in enumerate(axes.flat):
        if i < args.num_images:
            img = imgs[i].detach().cpu().numpy()
            if img.ndim == 3 and img.shape[0] in [1, 3]:
                img = np.transpose(img, (1, 2, 0))
            img = (img - img.min()) / (img.max() - img.min() + 1e-8)
            ax.imshow(img)
            ax.set_title(f"Image {i}", fontsize=10)
        ax.axis("off")

    plt.tight_layout()
    plt.savefig(output_path, dpi=args.dpi, bbox_inches="tight")
    plt.close(fig)

    logger.info(f"✅ Saved image grid to {output_path}")

    # Optional preview (if running interactively)
    if sys.stdout.isatty():
        plt.imshow(imgs[0].detach().cpu().numpy())
        plt.show()


if __name__ == "__main__":
    main()
