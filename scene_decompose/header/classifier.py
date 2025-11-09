import os
from pathlib import Path
from typing import Optional, Callable, Tuple

from gsplat_ext.utils.primitives import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import torchvision.transforms as T

from jafar import load_model


import torchvision.transforms as T
from torchvision.transforms import InterpolationMode

mean = (0.485, 0.456, 0.406)
std = (0.229, 0.224, 0.225)

# For images (PIL -> Tensor -> Normalize)
IMAGE_TRANSFORM = T.Compose(
    [
        T.Resize((448, 448), interpolation=InterpolationMode.BICUBIC),
        T.ToTensor(),  # makes [C,H,W] in [0,1]
        T.Normalize(mean=mean, std=std),
    ]
)

# For masks (PIL -> Tensor of longs). Use nearest to keep class ids intact.
MASK_TRANSFORM = T.Compose(
    [
        T.Resize((448, 448), interpolation=InterpolationMode.NEAREST),
        T.PILToTensor(),  # uint8 [1,H,W]
        T.Lambda(lambda x: x.squeeze(0).long()),  # [H,W] long
    ]
)


# ---------------------------
# Config
# ---------------------------
class Cfg:
    ade_root = "/data0/ADEChallengeData2016"  # set this!
    split = "training"  # "training" or "validation"
    val_split = "validation"
    num_classes = 151  # ADE20K has 150 semantic classes; 255 is 'void' to ignore
    batch_size = 2
    workers = 2
    epochs = 6
    lr = 3e-7
    weight_decay = 0.0
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # If your backbone needs specific image size/normalization, set them here:
    img_size = 518  # example; set as your backbone expects (or None to keep native)
    mean = (0.485, 0.456, 0.406)  # typical ImageNet
    std = (0.229, 0.224, 0.225)
    backbone = "vit_large_patch16_dinov3.lvd1689m"
    model_path = "vit_large_patch16_dinov3.lvd1689m.pth"


# ---------------------------
# ADE20K Dataset
# Folder structure:
#   ADEChallengeData2016/
#     images/{training,validation}/*.jpg
#     annotations/{training,validation}/*.png
# ---------------------------
class ADE20KDataset(Dataset):
    def __init__(
        self,
        root: str,
        split: str = "training",
        return_paths: bool = False,
    ):
        self.root = Path(root)
        self.img_dir = self.root / "images" / split
        self.ann_dir = self.root / "annotations" / split
        self.return_paths = return_paths

        self.ids = sorted([p.stem for p in self.img_dir.glob("*.jpg")])
        # Ensure matching masks exist
        self.ids = [i for i in self.ids if (self.ann_dir / f"{i}.png").exists()]

        # Default transform: ToTensor + normalize (tweak for your backbone)
        self.img_transform = IMAGE_TRANSFORM
        self.mask_transform = MASK_TRANSFORM

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx: int):
        id_ = self.ids[idx]
        img_path = self.img_dir / f"{id_}.jpg"
        ann_path = self.ann_dir / f"{id_}.png"

        img_pil = Image.open(img_path).convert("RGB")
        mask_pil = Image.open(ann_path)  # 0..149; 255 = ignore

        img = self.img_transform(img_pil)  # [3, H, W] float32
        mask = self.mask_transform(mask_pil)  # [H, W]   int64

        if self.return_paths:
            return img, mask, str(img_path)
        return img, mask


# ---------------------------
# Patch-level head (1x1 conv)
# ---------------------------
class PatchClassifierHead(nn.Module):
    def __init__(
        self,
        in_channels: int,
        num_classes: int,
        dropout_p: float = 0.0,
        use_bn: bool = False,
    ):
        super().__init__()
        layers = []
        if use_bn:
            layers.append(nn.BatchNorm2d(in_channels))
        if dropout_p > 0:
            layers.append(nn.Dropout2d(dropout_p))
        layers.append(nn.Conv2d(in_channels, num_classes, kernel_size=1, bias=True))
        self.head = nn.Sequential(*layers)

    def forward(self, feats: torch.Tensor) -> torch.Tensor:
        # feats: [B, C, Hf, Wf]
        return self.head(feats)


# ---------------------------
# Backbone placeholder
# Replace this with your real DINOv3 backbone wrapper
# It must return features shaped [B, C, Hf, Wf]
# ---------------------------
class Dinov3Backbone(nn.Module):
    def __init__(self, model: torch.nn.Module, backbone: torch.nn.Module):
        super().__init__()
        self.model = model
        self.backbone = backbone

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B,3,H,W] preprocessed as your backbone expects
        # Return dummy features here; replace with real backbone.forward()
        B, _, H, W = x.shape
        lr_feats, _ = self.backbone(x)
        hr_feats = self.model(
            x, lr_feats, (448, 448)
        )  # expect (B, C, H', W') or similar

        # Normalize the C channel (feature dimension) to unit norm per spatial location
        # hr_feats: [B, C, H', W']
        hr_feats = F.normalize(hr_feats, p=2, dim=1)
        return hr_feats


# ---------------------------
# Utils
# ---------------------------
def resize_mask_to_feature(
    mask: torch.Tensor, size_hw: Tuple[int, int]
) -> torch.Tensor:
    """Resize integer mask [B,H,W] to feature size using nearest neighbor."""
    # Convert to [B,1,H,W] -> interpolate (nearest) -> [B,Hf,Wf]
    mask = mask.unsqueeze(1).float()
    mask = F.interpolate(mask, size=size_hw, mode="nearest")
    return mask[:, 0].long()


def set_requires_grad(model: nn.Module, flag: bool):
    for p in model.parameters():
        p.requires_grad = flag


# ---------------------------
# Train / Eval
# ---------------------------
def train_one_epoch(backbone, head, loader, opt, device):
    backbone.eval()  # keep frozen
    head.train()

    total_loss, total_pix, total_correct = 0.0, 0, 0
    for imgs, masks in tqdm(loader, desc="Training"):
        imgs = imgs.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)  # [B,H,W], 0..149, void=255

        with torch.no_grad():
            feats = backbone(imgs)  # [B,C,Hf,Wf]
        B, C, Hf, Wf = feats.shape

        # Downsample mask to feature grid
        masks_ds = resize_mask_to_feature(masks, (Hf, Wf))  # [B,Hf,Wf]

        logits = head(feats)  # [B,num_classes,Hf,Wf]

        loss = F.cross_entropy(logits, masks_ds, ignore_index=255)

        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()

        total_loss += loss.item() * B

        # Compute train pixel acc (ignore void)
        with torch.no_grad():
            preds = logits.argmax(1)  # [B,Hf,Wf]
            valid = masks_ds != 255
            total_pix += valid.sum().item()
            total_correct += (preds.eq(masks_ds) & valid).sum().item()

    avg_loss = total_loss / len(loader.dataset)
    acc = total_correct / max(1, total_pix)
    return avg_loss, acc


@torch.no_grad()
def evaluate(backbone, head, loader, device):
    backbone.eval()
    head.eval()

    total_loss, total_pix, total_correct = 0.0, 0, 0
    for imgs, masks in loader:
        imgs = imgs.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)

        feats = backbone(imgs)
        B, C, Hf, Wf = feats.shape
        masks_ds = resize_mask_to_feature(masks, (Hf, Wf))

        logits = head(feats)
        loss = F.cross_entropy(logits, masks_ds, ignore_index=255)

        total_loss += loss.item() * B

        preds = logits.argmax(1)
        valid = masks_ds != 255
        total_pix += valid.sum().item()
        total_correct += (preds.eq(masks_ds) & valid).sum().item()

    avg_loss = total_loss / len(loader.dataset)
    acc = total_correct / max(1, total_pix)
    return avg_loss, acc


# ---------------------------
# Main
# ---------------------------
def main():
    device = torch.device(Cfg.device)

    # Datasets
    train_set = ADE20KDataset(root=Cfg.ade_root, split=Cfg.split)
    val_set = ADE20KDataset(root=Cfg.ade_root, split=Cfg.val_split)

    train_loader = DataLoader(
        train_set,
        batch_size=Cfg.batch_size,
        shuffle=True,
        num_workers=Cfg.workers,
        pin_memory=True,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_set,
        batch_size=Cfg.batch_size,
        shuffle=False,
        num_workers=Cfg.workers,
        pin_memory=True,
    )

    # Models
    project_root = os.getcwd()
    model, backbone = load_model(
        Cfg.backbone, project_root, Cfg.model_path
    )  # tuple of model and backbone
    backbone = Dinov3Backbone(model, backbone).to(device)
    set_requires_grad(backbone, False)  # freeze backbone

    head = PatchClassifierHead(
        in_channels=1024, num_classes=Cfg.num_classes, dropout_p=0.0, use_bn=False
    ).to(device)

    # INSERT_YOUR_CODE
    # Load head weights from checkpoint if available
    ckpt_path = "best_patch_head.pth"
    if os.path.exists(ckpt_path):
        checkpoint = torch.load(ckpt_path, map_location=device)
        if "head" in checkpoint:
            head.load_state_dict(checkpoint["head"])
            print(f"Loaded head weights from {ckpt_path}")
        else:
            print(f"Checkpoint {ckpt_path} does not contain 'head' key.")

    # Optimizer (head only)
    opt = torch.optim.AdamW(head.parameters(), lr=Cfg.lr, weight_decay=Cfg.weight_decay)

    # Train
    best_val = 0.0
    for epoch in range(1, Cfg.epochs + 1):
        tr_loss, tr_acc = train_one_epoch(backbone, head, train_loader, opt, device)
        va_loss, va_acc = evaluate(backbone, head, val_loader, device)

        print(
            f"[{epoch:03d}] train loss {tr_loss:.4f} acc {tr_acc:.4f} | "
            f"val loss {va_loss:.4f} acc {va_acc:.4f}"
        )

        if va_acc > best_val:
            best_val = va_acc
            torch.save({"head": head.state_dict()}, "best_patch_head.pth")


if __name__ == "__main__":
    main()
