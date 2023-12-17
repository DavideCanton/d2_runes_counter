# d2_runes_counter

Diablo 2 runes counter from storage screenshots.

Usage: `detector.py [-h] [-r RUNE_FOLDER] [-t THRESHOLD] [-v] image [image ...]`

Arguments:

```
positional arguments:
  image                 The screenshots. Supported resolutions are: 1920x1080.

options:
  -h, --help            show this help message and exit
  -r RUNE_FOLDER, --rune_folder RUNE_FOLDER
                        The folder where the rune images are stored. Defaults to "./runes".
  -t THRESHOLD, --threshold THRESHOLD
                        The minimum threshold for a match to be considered valid. The score is returned by the
                        metrics.structural_similarity function of skimage. Defaults to 0.3.
  -v, --verbose         Enable verbose output.
```
