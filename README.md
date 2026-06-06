# Human4D Combined: Body + Hands Reconstruction

A self-contained, minimal script that combines:

- **4D-Humans (HMR2)** for body pose estimation  
- **HaMeR** for hand pose estimation  
- **SMPL-X** for a unified body + hand mesh output  

Given a single RGB image, the script detects people, estimates body and hand poses, and fuses them into SMPL-X meshes.

## Output

For every detected person, the script writes:

- `person_{i}_smplx.obj` – SMPL-X mesh with HMR2 body + HaMeR hands
- `person_{i}_rendered.png` – rendered overlay on the input image

## Setup

### 1. Install Python dependencies

```bash
pip install -r requirements.txt
```

> **Note:** `detectron2` and `mmpose` often need platform-specific builds. If the pip install above fails for those packages, follow the official install guides:
> - [Detectron2](https://github.com/facebookresearch/detectron2/blob/main/INSTALL.md)
> - [MMPose](https://mmpose.readthedocs.io/en/latest/installation.html)

### 2. Download model checkpoints

Run the upstream fetch scripts (or manually download) so the following checkpoints exist:

```
_DATA/
├── hmr2/
│   └── logs/train/multiruns/hmr2/0/checkpoints/epoch=35-step=1000000.ckpt
├── hamer/
│   └── logs/train/multiruns/hamer/0/checkpoints/epoch=35-step=1000000.ckpt
└── vitpose_ckpts/
    └── vitpose+_huge/
        └── wholebody.pth
```

The default `CACHE_DIR_HAMER` is `./_DATA` (relative to this repo root). You can override it:

```bash
export HAMER_CACHE_DIR=/your/custom/cache/path
```

### 3. Download SMPL-X

Download the **SMPL-X neutral model** (`SMPLX_NEUTRAL_py3.pkl`) from the [SMPL-X website](https://smpl-x.is.tue.mpg.de/) and place it somewhere accessible.

## Usage

```bash
export SMPLX_PATH=/path/to/SMPLX_NEUTRAL_py3.pkl
python demo.py --img /path/to/image.jpg --out output
```

Optional arguments:

- `--device cuda` (default) or `--device cpu`
- `--out` output directory (default: `output`)

## Rendering note

Rendering uses OSMesa for headless/off-screen rendering. If you encounter shader errors, make sure the OSMesa shared library is available in `LD_LIBRARY_PATH`:

```bash
export LD_LIBRARY_PATH=/path/to/osmesa/lib:$LD_LIBRARY_PATH
```

On the original setup this was satisfied by the `osmesa_tmp` directory from the 4D-Humans repo.

## Credits

- [4D-Humans / HMR2](https://github.com/shubham-goel/4D-Humans)
- [HaMeR](https://github.com/geopavlakos/hamer)
- [SMPL-X](https://smpl-x.is.tue.mpg.de/)
