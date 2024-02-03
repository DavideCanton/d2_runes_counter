import argparse
from collections import Counter
from pathlib import Path

from src.detector import RESOLUTION, Row, RuneDetector


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
    res = ", ".join(f"{w}x{h}" for w, h in RESOLUTION)
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

        detector = RuneDetector(runes_folder, img_path, verbose=args.verbose)
        found = detector.detect_runes()
        order.extend(found)
        runes_cnt += Counter(r.runes[0].rune for r in found if r.runes[0].score > threshold)

    for row in order:
        print(f"({row.row:>2}, {row.col:>2}) -> ", end="")
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
