"""Fine-tune SegFormer on CARLA RGB/semantic-camera pairs for aerial BEV mapping.

The semantic camera is used only as offline supervision. The exported model
predicts from RGB at inference time and can be used by the fusion script.
"""

import argparse
import json
import os
import random
from pathlib import Path

os.environ.setdefault("HF_HOME", str(Path.cwd() / ".cache" / "huggingface"))

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from transformers import SegformerForSemanticSegmentation, SegformerImageProcessor


CLASS = {"unknown": 0, "road": 1, "vegetation": 2, "building": 3, "vehicle": 4, "person": 5}
CARLA_LABELS = {
    "road": (1, 6, 7, 8, 14, 16, 20, 24),
    "vegetation": (9, 22),
    "building": (2, 3, 5, 11, 12, 15, 17, 18, 19),
    "vehicle": (10,),
    "person": (4, 25),
}
MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)
IGNORE_INDEX = 255


def local_model_path(model_name: str) -> str:
    candidate = Path(model_name)
    if candidate.is_dir():
        return str(candidate)
    cache = Path(os.environ["HF_HOME"]) / "hub" / f"models--{model_name.replace('/', '--')}" / "snapshots"
    snapshots = sorted(path for path in cache.glob("*") if path.is_dir())
    if not snapshots:
        raise FileNotFoundError(f"Model is not cached locally: {model_name}")
    return str(snapshots[-1])


def read_mask(path: Path) -> np.ndarray:
    raw_tags = np.asarray(Image.open(path).convert("RGB"))[..., 0]
    mask = np.full(raw_tags.shape, IGNORE_INDEX, dtype=np.uint8)
    for name, tags in CARLA_LABELS.items():
        mask[np.isin(raw_tags, tags)] = CLASS[name]
    return mask


class CarlaSemanticDataset(Dataset):
    def __init__(self, root: Path, frame_ids: list[str], height: int, width: int) -> None:
        self.root, self.frame_ids, self.height, self.width = root, frame_ids, height, width

    def __len__(self) -> int:
        return len(self.frame_ids)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        frame_id = self.frame_ids[index]
        image = np.asarray(Image.open(self.root / "camera" / f"{frame_id}.png").convert("RGB"))
        mask = read_mask(self.root / "semantic" / f"{frame_id}.png")
        image = cv2.resize(image, (self.width, self.height), interpolation=cv2.INTER_LINEAR)
        mask = cv2.resize(mask, (self.width, self.height), interpolation=cv2.INTER_NEAREST)
        image = (image.astype(np.float32) / 255.0 - MEAN) / STD
        return torch.from_numpy(image.transpose(2, 0, 1)), torch.from_numpy(mask.astype(np.int64))


def confusion_matrix(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    prediction = logits.argmax(dim=1)
    valid = labels != IGNORE_INDEX
    values = labels[valid] * len(CLASS) + prediction[valid]
    return torch.bincount(values, minlength=len(CLASS) ** 2).reshape(len(CLASS), len(CLASS))


def scores(confusion: torch.Tensor) -> dict:
    diagonal = confusion.diag().float()
    denominator = confusion.sum(1).float() + confusion.sum(0).float() - diagonal
    iou = torch.where(denominator > 0, diagonal / denominator, torch.nan)
    present_in_validation = confusion.sum(1) > 0
    return {
        "mean_iou": round(float(torch.nanmean(iou[present_in_validation])), 4),
        "per_class_iou": {name: round(float(iou[class_id]), 4) if torch.isfinite(iou[class_id]) else None for name, class_id in CLASS.items()},
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Fine-tune a local SegFormer on CARLA aerial semantic labels.")
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--output-dir", default="outputs/models/segformer_carla_aerial")
    parser.add_argument("--base-model", default="nvidia/segformer-b0-finetuned-ade-512-512")
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--height", type=int, default=288)
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--person-weight", type=float, default=8.0, help="Cross-entropy weight for sparse pedestrian pixels.")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if args.height % 32 or args.width % 32:
        raise ValueError("--height and --width must be multiples of 32.")

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    root, output = Path(args.dataset), Path(args.output_dir)
    frame_ids = sorted(path.stem for path in (root / "camera").glob("*.png") if (root / "semantic" / path.name).exists())
    if len(frame_ids) < 10:
        raise RuntimeError("At least ten matched camera/semantic frames are required.")
    validation = [frame_id for index, frame_id in enumerate(frame_ids) if index % 5 == 0]
    training = [frame_id for index, frame_id in enumerate(frame_ids) if index % 5 != 0]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"train={len(training)} val={len(validation)} device={device}")

    train_loader = DataLoader(CarlaSemanticDataset(root, training, args.height, args.width), batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(CarlaSemanticDataset(root, validation, args.height, args.width), batch_size=args.batch_size)
    labels = {index: name for name, index in CLASS.items()}
    base_model = local_model_path(args.base_model)
    model = SegformerForSemanticSegmentation.from_pretrained(
        base_model,
        num_labels=len(CLASS),
        id2label=labels,
        label2id=CLASS,
        ignore_mismatched_sizes=True,
        local_files_only=True,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=0.01)
    class_weights = torch.tensor([0.0, 1.0, 1.0, 1.0, 0.0, args.person_weight], device=device)
    best_iou, best_state, history = -1.0, None, []

    for epoch in range(1, args.epochs + 1):
        model.train()
        losses = []
        for image, mask in train_loader:
            image, mask = image.to(device), mask.to(device)
            logits = model(pixel_values=image).logits
            logits = F.interpolate(logits, size=mask.shape[-2:], mode="bilinear", align_corners=False)
            loss = F.cross_entropy(logits, mask, ignore_index=IGNORE_INDEX, weight=class_weights)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            losses.append(float(loss.detach().cpu()))

        model.eval()
        matrix = torch.zeros((len(CLASS), len(CLASS)), dtype=torch.long)
        with torch.inference_mode():
            for image, mask in val_loader:
                image, mask = image.to(device), mask.to(device)
                logits = model(pixel_values=image).logits
                logits = F.interpolate(logits, size=mask.shape[-2:], mode="bilinear", align_corners=False)
                matrix += confusion_matrix(logits.cpu(), mask.cpu())
        metrics = scores(matrix)
        item = {"epoch": epoch, "train_loss": round(float(np.mean(losses)), 5), **metrics}
        history.append(item)
        print(json.dumps(item, ensure_ascii=False))
        if metrics["mean_iou"] > best_iou:
            best_iou = metrics["mean_iou"]
            best_state = {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}

    output.mkdir(parents=True, exist_ok=True)
    model.load_state_dict(best_state)
    model.save_pretrained(output)
    processor = SegformerImageProcessor.from_pretrained(base_model, local_files_only=True)
    processor.size = {"height": args.height, "width": args.width}
    processor.save_pretrained(output)
    (output / "training_report.json").write_text(json.dumps({"train_frames": training, "val_frames": validation, "history": history}, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"best model saved to {output} with mIoU={best_iou:.4f}")


if __name__ == "__main__":
    main()
