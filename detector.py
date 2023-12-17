import argparse
from collections import Counter
from collections.abc import Callable
from operator import itemgetter
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps
from skimage import metrics


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
}


class RuneDetector:
    def _load_runes(self, runes_folder: Path, res: ResDef) -> None:
        runes = []
        size = tuple(res.size)

        for r in runes_folder.glob("*.png"):
            with Image.open(r) as im:
                g = ImageOps.grayscale(im).resize(size)  # type: ignore
                img = np.array(g)
                runes.append((r.stem, img))

        self._runes = tuple(runes)

    def _load_image(self, image_path: Path) -> tuple[ResDef, np.ndarray]:
        with Image.open(image_path) as im:
            g = ImageOps.grayscale(im)
            img = np.array(g)
            res: tuple[int, int] = img.T.shape  # type: ignore
            return _RES[res], img

    def _detect(
        self, res: ResDef, img: np.ndarray, start: np.ndarray, threshold: float
    ) -> list[tuple[str, float]]:
        runes = self._runes

        end = start + res.size
        tst = img[start[0] : end[0], start[1] : end[1]]
        if (tst.shape != res.size).any():  # type: ignore
            return []

        imgs = []
        for n, r in runes:
            sc = metrics.structural_similarity(tst, r)
            if sc < threshold:
                continue

            imgs.append((n, sc))

        return imgs

    def _logFn(self, verbose: bool) -> Callable[[str], None]:
        if verbose:
            return print
        else:
            return lambda _: None

    def count_runes(
        self, image_path: Path, runes_folder: Path, threshold: float, verbose: bool
    ) -> dict[str, int]:
        log = self._logFn(verbose)
        res, img = self._load_image(image_path)
        self._load_runes(runes_folder, res)

        runes_cnt = Counter()
        by_score = itemgetter(1)

        start_row = res.start.copy()
        while start_row[0] <= res.end[0]:
            cur = start_row.copy()
            while cur[1] <= res.end[1]:
                log(f"Detecting at {cur}...")
                try_runes = self._detect(res, img, cur, threshold)
                if not try_runes:
                    log("Nothing, continue...")
                else:
                    rune, sc = max(try_runes, key=by_score)
                    log(f"Detected {rune} with score of {sc}...")
                    runes_cnt[rune] += 1
                cur[1] += res.size[1] + res.border[1]

            start_row[0] += res.size[0] + res.border[0]

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
        default=0.3,
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

    for image in args.image:
        img_path = Path(image)
        if not img_path.exists():
            raise FileNotFoundError(img_path)

        detector = RuneDetector()
        runes_cnt += detector.count_runes(
            img_path, runes_folder, args.threshold, verbose=args.verbose
        )

    for r, v in sorted(runes_cnt.items(), key=lambda t: t[1], reverse=True):
        print(f"{r.title()}: {v}")
    print(f"Total: {sum(runes_cnt.values())}")


if __name__ == "__main__":
    main()
