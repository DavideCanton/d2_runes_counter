from dataclasses import dataclass
from operator import attrgetter
from pathlib import Path

import numpy as np
import numpy.typing as npt
from PIL import Image, ImageOps
from skimage import metrics, morphology


class ResolutionSizes:
    def __init__(
        self,
        size: tuple[int, int],
        border: tuple[int, int],
        start: tuple[int, int],
        end: tuple[int, int],
    ):
        self.size = np.array(size, dtype=np.uint)
        self.border = np.array(border, dtype=np.uint)
        self.start = np.array(start, dtype=np.uint)
        self.end = np.array(end, dtype=np.uint)


RESOLUTION: dict[tuple[int, int], ResolutionSizes] = {
    (1920, 1080): ResolutionSizes((45, 45), (4, 4), (224, 165), (708, 651)),
    (3840, 2160): ResolutionSizes((91, 91), (7, 7), (447, 329), (1419, 1301)),
}

_BIN_THRESHOLD = 50


@dataclass(order=True)
class RuneMatch:
    score: float
    rune: str


@dataclass
class Row:
    row: int
    col: int
    runes: list[RuneMatch]


_Image = npt.NDArray[np.uint8]
M = np.iinfo(np.uint8).max


class RuneDetector:
    _runes: dict[str, _Image]
    _mask: npt.NDArray[np.ubyte]
    _res: ResolutionSizes
    _img: _Image
    _verbose: bool

    def __init__(self, runes_folder: Path, image_path: Path, verbose: bool) -> None:
        res, img = self._load_image(image_path)

        self._res = res
        self._runes = {}
        self._img = img
        self._verbose = verbose

        self._load_runes(runes_folder)

    def _load_runes(self, runes_folder: Path) -> None:
        size = tuple(self._res.size)

        with Image.open(runes_folder / "_mask.png") as mask:
            mask = mask.resize(size)  # type: ignore

        self._mask = np.clip(np.array(mask), 0, 1)

        for rune_path in runes_folder.glob("*.png"):
            if rune_path.stem == "_mask":
                continue

            with Image.open(rune_path) as im:
                g = ImageOps.grayscale(im)

            rune_img = np.array(g.resize(size), dtype=np.uint8)  # type: ignore
            rune_img = self._binarize(rune_img)
            rune_img = self._apply_mask(rune_img)
            self._runes[rune_path.stem] = rune_img

    def _binarize(self, img: _Image) -> _Image:
        return np.where(img < _BIN_THRESHOLD, 0, M)

    def _apply_mask(self, img: _Image) -> _Image:
        footprint = morphology.disk(2)
        img = img - morphology.white_tophat(img, footprint)  # type: ignore
        return np.where(self._mask == 0, M, img)

    def _load_image(self, image_path: Path) -> tuple[ResolutionSizes, _Image]:
        with Image.open(image_path) as im:
            g = ImageOps.grayscale(im)
            img = np.array(g)
            img = self._binarize(img)
            res: tuple[int, int] = img.T.shape  # type: ignore
            if resDef := RESOLUTION[res]:
                return resDef, img
            raise ValueError(f"Unhandled resolution {res}")

    def _single_detect(self, start: np.ndarray) -> list[RuneMatch]:
        runes = self._runes
        res = self._res
        img = self._img

        end = start + res.size
        self._log(f"Detecting at {start} -> {end-1}...", end=" ")
        img_part = img[start[0] : end[0], start[1] : end[1]]
        if (img_part.shape != res.size).any():  # type: ignore
            return []

        img_part = self._apply_mask(img_part)

        imgs = []
        for name, rune_img in runes.items():
            score = metrics.structural_similarity(
                img_part, rune_img, win_size=3, data_range=rune_img.max() - rune_img.min()
            )
            imgs.append(RuneMatch(score, name))

        return imgs

    def _log(self, *a, **kw):
        if self._verbose:
            print(*a, **kw)

    def detect_runes(self) -> list[Row]:
        runes_cnt = []
        by_score = attrgetter("score")
        res = self._res

        start_row = res.start.copy()
        row, col = 0, 0

        while start_row[0] <= res.end[0]:
            cur = start_row.copy()

            while cur[1] <= res.end[1]:
                try_runes = self._single_detect(cur)
                matches = sorted(try_runes, key=by_score, reverse=True)[:5]
                best = matches[0]
                self._log(f"Detected {best.rune} with score of {best.score}...")
                runes_cnt.append(Row(row, col, matches))
                cur[1] += res.size[1] + res.border[1]
                col += 1

            start_row[0] += res.size[0] + res.border[0]
            row += 1
            col = 0

        return runes_cnt
