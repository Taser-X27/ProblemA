"""Project-local ViT sampling, paired augmentation and metrics."""
import hashlib
import json
import math
import multiprocessing as mp
from pathlib import Path
import random
import re
import numpy as np
from PIL import Image, ImageOps
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from .common import ROOT,sha,data_root,inventory,read


def inputs(override=None,seed=42,splits=('train','validation')):
    root=data_root(override)
    indexed,classes=inventory(override,splits)
    records,provenance,protected={},{},{}
    for split,rows in indexed.items():
        records[split]=[]
        provenance[split]={r['file']:r for r in read(root/split/'provenance.json')['samples']}
        for row in rows:
            local_seed=int.from_bytes(hashlib.sha256(f"{seed}:{row['sample_id']}".encode()).digest()[:8],'big')
            _,indices=sample_frames(list(range(row['frames'])),16,local_seed)
            records[split].append(dict(sample_id=row['sample_id'],label=row['label'],split=split,
                available_frames=row['frames'],selected_indices_zero_based=indices))
        for name in ['manifest.json','provenance.json']:
            path=root/split/name;protected[str(path)]=sha(path)
    return root,records,provenance,classes,protected

def sample_frames(paths,count,seed):
    if len(paths)<count:
        indices=list(range(len(paths)))+[len(paths)-1]*(count-len(paths))
    else:
        rng=random.Random(seed)
        indices=[rng.randrange(i*len(paths)//count,(i+1)*len(paths)//count) for i in range(count)]
    return [paths[i] for i in indices],indices

def frame_key(path):
    name = path.stem
    name = re.sub(r"^(Depth|IR)_", "", name)
    name = re.sub(r"_Color$", "", name)
    return name

def frame_number(path):
    match = re.search(r"_(\d+)(?:_Color)?\.png$", path.name)
    if not match:
        raise ValueError(f"Unexpected frame name: {path}")
    return int(match[1])

def compute_metrics(labels, predictions, classes):
    confusion = torch.zeros(40, 40, dtype=torch.int64)
    for label, prediction in zip(labels.tolist(), predictions.tolist()):
        confusion[label, prediction] += 1
    support = confusion.sum(1)
    predicted = confusion.sum(0)
    true_positive = confusion.diag()
    recall = true_positive.float() / support.clamp_min(1)
    precision = true_positive.float() / predicted.clamp_min(1)
    f1 = 2 * precision * recall / (precision + recall).clamp_min(1e-12)
    return {"samples": len(labels), "correct": int(true_positive.sum()),
            "accuracy": float((labels == predictions).float().mean()),
            "macro_f1": float(f1.mean()), "balanced_accuracy": float(recall.mean()),
            "per_class": [{"id": i, "name": name, "support": int(support[i]),
                           "precision": float(precision[i]), "recall": float(recall[i]), "f1": float(f1[i])}
                          for i, name in enumerate(classes)],
            "confusion_matrix_true_rows_predicted_columns": confusion.tolist()}

def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def schedule_factor(step, total, warmup):
    if warmup and step < warmup:
        return .01 + .99 * (step + 1) / warmup
    progress = min(1., max(0., (step - warmup) / max(1, total - warmup - 1)))
    return .01 + .99 * .5 * (1 + math.cos(math.pi * progress))

class Videos(Dataset):
    def __init__(self, records, data_root, mode, training, seed):
        self.records, self.training, self.seed = records, training, seed
        self.names = ["Depth_Color", "IR"] if mode == "joint" else ["IR" if mode == "ir" else "Depth_Color"]
        self.epoch = mp.Value("i", 0, lock=False)
        self.paths = []
        split = records[0].get("split", "train" if training else "validation")
        for row in records:
            paired = {name: sorted((data_root / split / row["sample_id"] / name).glob("*.png"), key=frame_number)
                      for name in ["Depth_Color", "IR"]}
            if not paired["IR"] or [frame_key(p) for p in paired["IR"]] != [frame_key(p) for p in paired["Depth_Color"]]:
                raise ValueError(f"Missing or unpaired images: {split}/{row['sample_id']}")
            if len(paired["IR"]) != row["available_frames"]:
                raise ValueError(f"Video frame count changed: {row['sample_id']}")
            self.paths.append(paired)
        self.mean = torch.tensor([.485, .456, .406]).view(3, 1, 1, 1)
        self.std = torch.tensor([.229, .224, .225]).view(3, 1, 1, 1)

    def __len__(self):
        return len(self.records)

    @staticmethod
    def crop_box(width, height, rng):
        for _ in range(10):
            area = width * height * rng.uniform(.7, 1.)
            ratio = math.exp(rng.uniform(math.log(.75), math.log(4 / 3)))
            cw, ch = round(math.sqrt(area * ratio)), round(math.sqrt(area / ratio))
            if 0 < cw <= width and 0 < ch <= height:
                left, top = rng.randrange(width - cw + 1), rng.randrange(height - ch + 1)
                return left, top, left + cw, top + ch
        side = min(width, height)
        left, top = (width - side) // 2, (height - side) // 2
        return left, top, left + side, top + side

    def __getitem__(self, index):
        row = self.records[index]
        local_seed = int.from_bytes(hashlib.sha256(f"{self.seed}:{self.epoch.value}:{row['sample_id']}".encode()).digest()[:8], "big")
        rng = random.Random(local_seed)
        if self.training:
            _, selected = sample_frames(self.paths[index]["IR"], 16, local_seed)
        else:
            selected = row["selected_indices_zero_based"]
        if len(selected) != 16 or not all(0 <= i < row["available_frames"] for i in selected):
            raise ValueError("Invalid saved frame sampling")
        views, box, flip = [], None, rng.random() < .5
        for name in self.names:
            images = []
            for i in selected:
                with Image.open(self.paths[index][name][i]) as source:
                    image = source.convert("RGB")
                if self.training:
                    if box is None:
                        box = self.crop_box(*image.size, rng)
                    image = image.crop(box).resize((224, 224), Image.Resampling.BICUBIC)
                    if flip:
                        image = ImageOps.mirror(image)
                else:
                    width, height = image.size
                    scale = 224 / min(width, height)
                    image = image.resize((round(width * scale), round(height * scale)), Image.Resampling.BICUBIC)
                    left, top = (image.width - 224) // 2, (image.height - 224) // 2
                    image = image.crop((left, top, left + 224, top + 224))
                images.append(np.array(image))
            values = torch.from_numpy(np.stack(images)).permute(3, 0, 1, 2).float().div_(255.)
            views.append(((values - self.mean) / self.std).contiguous())
        return index, torch.stack(views), row["label"]

def loader(dataset, batch, workers, generator=None, shuffle=False):
    return DataLoader(dataset, batch_size=batch, shuffle=shuffle, num_workers=workers,
                      generator=generator, pin_memory=True, persistent_workers=workers > 0)

def metrics_with_subjects(labels, logits, classes, records, provenance, smoothing):
    result = compute_metrics(labels, logits.argmax(1), classes)
    result["loss"] = float(nn.functional.cross_entropy(logits, labels))
    result["loss_smoothed"] = float(nn.functional.cross_entropy(logits, labels, label_smoothing=smoothing))
    subjects = {}
    for i, row in enumerate(records):
        name = provenance[row["sample_id"]]["subject"]
        group = subjects.setdefault(name, {"samples": 0, "correct": 0})
        group["samples"] += 1
        group["correct"] += int(logits[i].argmax() == labels[i])
    for group in subjects.values():
        group["accuracy"] = group["correct"] / group["samples"]
    result["per_subject"] = subjects
    return result
