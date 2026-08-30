# MAHPL aligned 2D-3D pavement dataset: 500-sample release

This repository releases 500 pixel-aligned pavement samples from the private
dataset used to evaluate Morphology-Aware Hierarchical Prototype Learning
(MAHPL). It is provided to support independent inspection and verification of
the private-dataset experiments reported in the associated manuscript.

## Contents

Each sample is identified by the same eight-digit ID in three directories:

```text
2D/DL2D<ID>.jpg       2D pavement texture image
3D/DL3D<ID>.bin       aligned 3D pavement surface array
Truth/DL2D<ID>.png    pixel-wise semantic label mask
```

The release contains 500 complete triplets (1,500 files total):

| Component | Count | Shape | Format |
|---|---:|---|---|
| 2D | 500 | 256 x 512 | JPEG; grayscale or RGB |
| 3D | 500 | 256 x 512 | little-endian IEEE-754 `float32`, row-major |
| Truth | 500 | 256 x 512 | RGB PNG with identical channels; values 0-8 |

Every `.bin` file contains exactly 131,072 finite `float32` values and can be
reshaped directly to `(256, 512)`. Across this release, the observed 3D value
range is `[-2.5, 2.5]`. The images contain no EXIF metadata.

## Semantic labels

| Value | Class |
|---:|---|
| 0 | background |
| 1 | crack |
| 2 | pothole |
| 3 | sealed crack |
| 4 | pavement marking |
| 5 | expansion joint |
| 6 | patch |
| 7 | manhole cover |
| 8 | scratch |

## Python loader

```python
from pathlib import Path

import numpy as np
from PIL import Image

root = Path(".")
sample_id = "00000009"

image = np.asarray(Image.open(root / "2D" / f"DL2D{sample_id}.jpg"))
surface = np.fromfile(
    root / "3D" / f"DL3D{sample_id}.bin", dtype="<f4"
).reshape(256, 512)
mask_rgb = np.asarray(Image.open(root / "Truth" / f"DL2D{sample_id}.png"))
mask = mask_rgb[..., 0]
```

## Dataset context

The complete private dataset contains 5,000 aligned samples acquired using a
high-resolution 3D laser imaging system from more than 20 asphalt-pavement
sections in China over four years. Each complete sample consists of a 2D
texture image, an aligned 3D pavement surface array, and a pixel-level semantic
mask. The eight foreground classes are listed above.

This repository contains a 500-sample release rather than the complete private
dataset. 



