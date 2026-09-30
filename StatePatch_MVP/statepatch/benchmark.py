from __future__ import annotations
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Tuple
import json
from PIL import Image, ImageDraw, ImageFont

W, H = 832, 480
BG = (224, 220, 210)
TABLE = (165, 132, 100)
BOX = (135, 104, 72)
OUTLINE = (40, 40, 40)


@dataclass
class BenchmarkItem:
    sample_id: str
    pair_id: str
    state_type: str
    state_value: str
    label: int
    image: str
    prompt: str
    seed: int


def _scene(state_type: str, value: str, path: Path):
    im = Image.new("RGB", (W, H), BG)
    d = ImageDraw.Draw(im)
    d.rectangle([0, 310, W, H], fill=TABLE)
    # open container
    cx, cy = W // 2, 250
    d.rectangle([cx - 145, cy - 70, cx + 145, cy + 70], fill=BOX, outline=OUTLINE, width=5)
    d.rectangle([cx - 155, cy - 120, cx + 155, cy - 70], fill=(155, 120, 82), outline=OUTLINE, width=5)

    if state_type == "color":
        color = (205, 45, 45) if value == "red" else (45, 75, 205)
        d.rectangle([cx - 45, cy - 40, cx + 45, cy + 50], fill=color, outline=OUTLINE, width=4)
    elif state_type == "position":
        x = cx - 75 if value == "left" else cx + 75
        d.ellipse([x - 38, cy - 38, x + 38, cy + 38], fill=(230, 185, 40), outline=OUTLINE, width=4)
    elif state_type == "count":
        xs = [cx] if value == "one" else [cx - 55, cx + 55]
        for x in xs:
            d.ellipse([x - 32, cy - 32, x + 32, cy + 32], fill=(70, 175, 90), outline=OUTLINE, width=4)
    elif state_type == "shape":
        if value == "circle":
            d.ellipse([cx - 45, cy - 45, cx + 45, cy + 45], fill=(145, 75, 185), outline=OUTLINE, width=4)
        else:
            d.polygon([(cx, cy - 55), (cx - 55, cy + 45), (cx + 55, cy + 45)], fill=(145, 75, 185), outline=OUTLINE)
    else:
        raise ValueError(state_type)
    im.save(path)


def make_statepatchbench(out_dir: str | Path, pairs_per_type: int = 8) -> List[BenchmarkItem]:
    out_dir = Path(out_dir)
    image_dir = out_dir / "images"
    image_dir.mkdir(parents=True, exist_ok=True)
    tasks = {
        "color": ("red", "blue"),
        "position": ("left", "right"),
        "count": ("one", "two"),
        "shape": ("circle", "triangle"),
    }
    prompt = (
        "Continue the exact same tabletop scene. The container closes, the camera slowly pans away "
        "for several seconds, then returns to the same container and opens it again. Preserve every "
        "persistent property of the scene and object from the initial image. Cinematic but physically "
        "consistent motion; do not introduce new objects."
    )
    items: List[BenchmarkItem] = []
    for state_type, (v0, v1) in tasks.items():
        for i in range(pairs_per_type):
            pair_id = f"{state_type}_{i:03d}"
            for label, value in enumerate((v0, v1)):
                sid = f"{pair_id}_{label}"
                rel = Path("images") / f"{sid}.png"
                _scene(state_type, value, out_dir / rel)
                items.append(BenchmarkItem(
                    sample_id=sid,
                    pair_id=pair_id,
                    state_type=state_type,
                    state_value=value,
                    label=label,
                    image=str(rel),
                    prompt=prompt,
                    seed=1000 + i,
                ))
    manifest = out_dir / "statepatchbench.jsonl"
    with manifest.open("w", encoding="utf-8") as f:
        for item in items:
            f.write(json.dumps(asdict(item)) + "\n")
    return items


def read_manifest(path: str | Path) -> List[Dict]:
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]
