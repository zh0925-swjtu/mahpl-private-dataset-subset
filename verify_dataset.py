from __future__ import annotations

import csv
import hashlib
from pathlib import Path

import numpy as np
from PIL import Image


ROOT = Path(__file__).resolve().parent
EXPECTED_COUNT = 500
EXPECTED_SHAPE = (256, 512)
EXPECTED_LABELS = set(range(9))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def ids(folder: str, pattern: str, prefix: str) -> set[str]:
    return {
        path.stem.removeprefix(prefix)
        for path in (ROOT / folder).glob(pattern)
    }


def main() -> None:
    ids_2d = ids("2D", "DL2D*.jpg", "DL2D")
    ids_3d = ids("3D", "DL3D*.bin", "DL3D")
    ids_truth = ids("Truth", "DL2D*.png", "DL2D")
    assert ids_2d == ids_3d == ids_truth, "The three modalities are not paired"
    assert len(ids_2d) == EXPECTED_COUNT, f"Expected {EXPECTED_COUNT} samples"

    observed_labels: set[int] = set()
    for sample_id in sorted(ids_2d):
        with Image.open(ROOT / "2D" / f"DL2D{sample_id}.jpg") as image:
            assert image.size == EXPECTED_SHAPE[::-1]

        surface = np.fromfile(ROOT / "3D" / f"DL3D{sample_id}.bin", dtype="<f4")
        assert surface.size == EXPECTED_SHAPE[0] * EXPECTED_SHAPE[1]
        assert np.isfinite(surface).all()

        with Image.open(ROOT / "Truth" / f"DL2D{sample_id}.png") as image:
            mask_rgb = np.asarray(image)
        assert mask_rgb.shape == (*EXPECTED_SHAPE, 3)
        assert np.array_equal(mask_rgb[..., 0], mask_rgb[..., 1])
        assert np.array_equal(mask_rgb[..., 0], mask_rgb[..., 2])
        observed_labels.update(int(value) for value in np.unique(mask_rgb[..., 0]))

    assert observed_labels == EXPECTED_LABELS, observed_labels

    manifest_path = ROOT / "manifest-sha256.csv"
    with manifest_path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == EXPECTED_COUNT * 3
    for row in rows:
        path = ROOT / Path(row["path"])
        assert path.is_file(), path
        assert path.stat().st_size == int(row["bytes"]), path
        assert sha256(path) == row["sha256"], path

    print(f"Verified {EXPECTED_COUNT} paired samples and {len(rows)} checksums.")


if __name__ == "__main__":
    main()

