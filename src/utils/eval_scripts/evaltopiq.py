import os
import argparse
import torch
import pyiqa
from PIL import Image
from torchvision import transforms


def load_image(path):
    transform = transforms.Compose([
        transforms.ToTensor()
    ])

    img = Image.open(path).convert("RGB")
    return transform(img).unsqueeze(0)


def run_topiq(folder, device="cuda", save_csv=None):

    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"

    print(f"Using device: {device}")

    # Same initialization as pipeline
    iqa = pyiqa.create_metric("topiq_nr", device=device)

    results = []

    files = sorted(os.listdir(folder))

    for file in files:

        if not file.lower().endswith((".jpg", ".jpeg", ".png")):
            continue

        path = os.path.join(folder, file)

        img = load_image(path).to(device)

        with torch.no_grad():
            score = iqa(img).item()

        print(f"{file:30s}  TOPIQ: {score:.4f}")

        results.append((file, score))

    # Save CSV if requested
    if save_csv:
        with open(save_csv, "w") as f:
            f.write("image,topiq_score\n")
            for name, score in results:
                f.write(f"{name},{score:.4f}\n")

        print(f"\nSaved results to {save_csv}")


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--folder", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--save_csv", default=None)

    args = parser.parse_args()

    run_topiq(args.folder, args.device, args.save_csv)