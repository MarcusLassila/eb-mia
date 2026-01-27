from accelerate.accelerate import AcceleratorLite
from generative_models.utils import load_model

import torch
import matplotlib.pyplot as plt
import argparse
from pathlib import Path

def plot_images(images, name="temp_image"):
    # Create the 4x4 grid
    fig, axes = plt.subplots(4, 4, figsize=(6, 6))
    axes = axes.flatten()

    for img, ax in zip(images, axes):
        img = img.permute(1, 2, 0)
        img = img.clamp(0, 1)
        ax.imshow(img)
        ax.axis("off")

    plt.tight_layout()
    Path("./images").mkdir(parents=True, exist_ok=True)
    plt.savefig(f"./images/{name}.png")
    plt.close(fig)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-size", default=16, type=int)
    parser.add_argument("--model-path", type=str, required=True)
    parser.add_argument("--seed", default=42, type=int)
    parser.add_argument("--save-raw-data", action="store_true")
    args = parser.parse_args()

    accelerator = AcceleratorLite(torch_compile=False, base_seed=args.seed)
    model_path = Path(args.model_path)
    model, _ = load_model(model_path, accelerator.device)
    batch_size = args.batch_size
    gen_batch = model.sample(batch_size).cpu() 
    if args.save_raw_data:
        Path("./images").mkdir(parents=True, exist_ok=True)
        torch.save(gen_batch, "./images/{model_path.stem}_image_raw.pth")
    else:
        plot_images(gen_batch, name=f"{model_path.stem}_images")
