from pathlib import Path

import requests

URL = "https://d2runewizard.com/_next/image?url=%2Fimages%2Frunes%2F{name}.png&w=96&q=100"


RUNES = [
    "el",
    "eld",
    "tir",
    "nef",
    "eth",
    "ith",
    "tal",
    "ral",
    "ort",
    "thul",
    "amn",
    "sol",
    "shael",
    "dol",
    "hel",
    "io",
    "lum",
    "ko",
    "fal",
    "lem",
    "pul",
    "um",
    "mal",
    "ist",
    "gul",
    "vex",
    "ohm",
    "lo",
    "sur",
    "ber",
    "jah",
    "cham",
    "zod",
]

DL_FOLDER = Path.cwd() / "img"


def main():
    DL_FOLDER.mkdir(parents=True, exist_ok=True)

    for r in RUNES:
        print(f"Rune {r}...")
        f = DL_FOLDER / f"{r}.png"
        if f.exists():
            print("Existing, continue")
            continue

        url = URL.format(name=r)
        print("Downloading...")
        data = requests.get(url).content

        with f.open("wb") as fo:
            fo.write(data)

        print("Saved...")


if __name__ == "__main__":
    main()
