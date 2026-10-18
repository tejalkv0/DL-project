"""Run inference with the saved CIFAR-10 ResNet-18 model.

Usage:
    python predict.py path/to/image.jpg
    python predict.py img1.png img2.png --topk 3 --model cifar10_resnet18.pth
"""
import argparse
import torch, torch.nn as nn
from PIL import Image
from torchvision import models, transforms


def load_model(path):
    ckpt = torch.load(path, map_location="cpu")
    model = models.resnet18(weights=None)
    model.fc = nn.Sequential(nn.Dropout(0.3), nn.Linear(512, len(ckpt["classes"])))
    model.load_state_dict(ckpt["state_dict"])
    return model.eval(), ckpt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("images", nargs="+")
    ap.add_argument("--model", default="cifar10_resnet18.pth")
    ap.add_argument("--topk", type=int, default=3)
    args = ap.parse_args()

    model, ckpt = load_model(args.model)
    tf = transforms.Compose([
        transforms.Resize(ckpt["img_size"]),
        transforms.ToTensor(),
        transforms.Normalize(ckpt["mean"], ckpt["std"]),
    ])
    for path in args.images:
        x = tf(Image.open(path).convert("RGB")).unsqueeze(0)
        with torch.no_grad():
            probs = torch.softmax(model(x), dim=1)[0]
        top = probs.topk(args.topk)
        print(f"\n{path}")
        for score, idx in zip(top.values, top.indices):
            print(f"  {ckpt['classes'][idx]:<12} {score.item()*100:5.1f}%")


if __name__ == "__main__":
    main()
