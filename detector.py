import argparse
from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass
from operator import attrgetter
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps
from skimage import metrics, morphology


class ResDef:
    def __init__(
        self,
        size: tuple[int, int],
        border: tuple[int, int],
        start: tuple[int, int],
        end: tuple[int, int],
    ):
        self.size = np.array(size)
        self.border = np.array(border)
        self.start = np.array(start)
        self.end = np.array(end)


_RES: dict[tuple[int, int], ResDef] = {
    (1920, 1080): ResDef((45, 45), (4, 4), (224, 165), (708, 651)),
    (3840, 2160): ResDef((91, 91), (7, 7), (447, 329), (1419, 1301)),
}

_BIN_THRESHOLD = 50


@dataclass(order=True)
class RuneMatch:
    score: float
    rune: str


@dataclass
class Row:
    i: int
    j: int
    runes: list[RuneMatch]


class RuneDetector:
    def _load_runes(self, runes_folder: Path, res: ResDef) -> None:
        runes = []
        size = tuple(res.size)

        with Image.open(runes_folder / "_mask.png") as m:
            self._mask = np.clip(np.array(m.resize(size)), 0, 1)  # type: ignore

        for r in runes_folder.glob("*.png"):
            if r.stem == "_mask":
                continue
            with Image.open(r) as im:
                g = ImageOps.grayscale(im).resize(size)  # type: ignore
                img = np.array(g)
                img = self._binarize(img)
                img = self._apply_mask(img)
                runes.append((r.stem, img))

        self._runes = tuple(runes)

    def _binarize(self, img):
        return np.where(img < _BIN_THRESHOLD, 0, 255)

    def _apply_mask(self, img):
        footprint = morphology.disk(2)
        img = img - morphology.white_tophat(img, footprint)
        return np.where(self._mask == 0, 255, img)

    def _load_image(self, image_path: Path) -> tuple[ResDef, np.ndarray]:
        with Image.open(image_path) as im:
            g = ImageOps.grayscale(im)
            img = np.array(g)
            img = self._binarize(img)
            res: tuple[int, int] = img.T.shape  # type: ignore
            if res not in _RES:
                raise ValueError(f"Unhandled resolution {res}")
            return _RES[res], img

    def _single_detect(
        self, res: ResDef, img: np.ndarray, start: np.ndarray, log: Callable
    ) -> list[RuneMatch]:
        runes = self._runes

        end = start + res.size
        log(f"Detecting at {start} -> {end-1}...", end=" ")
        tst = img[start[0] : end[0], start[1] : end[1]]
        if (tst.shape != res.size).any():  # type: ignore
            return []

        tst = self._apply_mask(tst)

        imgs = []
        for n, r in runes:
            sc = metrics.structural_similarity(tst, r, win_size=3, data_range=r.max() - r.min())
            imgs.append(RuneMatch(sc, n))

        return imgs

    def _logFn(self, verbose: bool) -> Callable[[str], None]:
        if verbose:
            return print
        else:
            return lambda *a, **kw: None

    def detect_runes(self, image_path: Path, runes_folder: Path, verbose: bool) -> list[Row]:
        log = self._logFn(verbose)
        res, img = self._load_image(image_path)
        self._load_runes(runes_folder, res)

        runes_cnt = []
        by_score = attrgetter("score")

        start_row = res.start.copy()
        i, j = 0, 0
        while start_row[0] <= res.end[0]:
            cur = start_row.copy()
            while cur[1] <= res.end[1]:
                try_runes = self._single_detect(res, img, cur, log=log)
                r = sorted(try_runes, key=by_score, reverse=True)[:5]
                log(f"Detected {r[0].rune} with score of {r[0].score}...")
                runes_cnt.append(Row(i, j, r))
                cur[1] += res.size[1] + res.border[1]
                j += 1

            start_row[0] += res.size[0] + res.border[0]
            i += 1
            j = 0

        return runes_cnt


def create_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-r",
        "--rune_folder",
        default="./runes",
        help='The folder where the rune images are stored. Defaults to "./runes".',
    )
    parser.add_argument(
        "-t",
        "--threshold",
        default=0.7,
        type=float,
        help=(
            "The minimum threshold for a match to be considered valid. "
            "The score is returned by the metrics.structural_similarity function of skimage. "
            "Defaults to 0.3."
        ),
    )
    parser.add_argument(
        "-v", "--verbose", default=False, action="store_true", help="Enable verbose output."
    )
    res = ", ".join(f"{w}x{h}" for w, h in _RES)
    parser.add_argument(
        "image", nargs="+", help=f"The screenshots. Supported resolutions are: {res}."
    )
    return parser


def main():
    parser = create_parser()
    args = parser.parse_args()

    runes_folder = Path(args.rune_folder)
    if not runes_folder.is_dir():
        raise ValueError(f"{runes_folder} is not a directory")

    runes_cnt = Counter()
    order: list[Row] = []
    threshold = args.threshold

    for image in args.image:
        img_path = Path(image)
        if not img_path.exists():
            raise FileNotFoundError(img_path)

        detector = RuneDetector()
        found = detector.detect_runes(img_path, runes_folder, verbose=args.verbose)
        order.extend(found)
        runes_cnt += Counter(r.runes[0].rune for r in found if r.runes[0].score > threshold)

    for row in order:
        print(f"({row.i:>2}, {row.j:>2}) -> ", end="")
        for r in row.runes:
            print(f"{r.rune:>5} [{r.score:.2f}]", end=", ")

        if row.runes[0].score > threshold:
            print("OK")
        else:
            print("NO")

    for r, v in sorted(runes_cnt.items(), key=lambda t: t[1], reverse=True):
        if r is None:
            print(f"**UNKNOWN**: {v}")
        else:
            print(f"{r.title()}: {v}")
    print(f"Total: {sum(runes_cnt.values())}")


if __name__ == "__main__":
    main()
