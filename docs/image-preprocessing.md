# Grayscale image preprocessing

Run this command from the repository root after installing the Python dependencies. It downloads a pinned public archive and writes normalized arrays, a sample image grid, and split summaries under `.work/medical/`.

```sh
python -m label_lookup medical
```

The medical command processed all 5,856 [PneumoniaMNIST images](https://medmnist.com/v2) using the supplied splits, with 4,708 training, 524 validation, and 624 test images. It verified dimensions and labels and produced normalized 28×28 grayscale arrays, a sample grid, and counts per split. The summary, grid, and arrays were byte-identical on a rerun. The [official archive](https://zenodo.org/records/10519652) is checksum-verified before processing.
