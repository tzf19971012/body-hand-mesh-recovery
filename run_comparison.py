#!/usr/bin/env python3
"""
Run baseline 4D-Humans and optimized human4d_combined demos side-by-side
on hand-visible test images, then create comparison visualizations.
"""

import os
import sys
import shutil
import subprocess
from pathlib import Path
import cv2
import numpy as np

BASELINE_REPO = Path("/home/tianzefan/human4d_mano/4D-Humans")
OPTIMIZED_REPO = Path("/home/tianzefan/human4d_mano/human4d_combined")
SMPLX_PATH = "/mnt/d/download/SMPLX_NEUTRAL_py3.pkl"
OSMESA_LIB = "/home/tianzefan/human4d_mano/4D-Humans/osmesa_tmp/usr/lib/x86_64-linux-gnu"
HAMER_CACHE_DIR = "/mnt/d/download/hamer_demo_data/_DATA"

INPUT_DIR = OPTIMIZED_REPO / "comparison_inputs"
BASE_OUT = OPTIMIZED_REPO / "comparison_outputs" / "baseline"
OPT_OUT = OPTIMIZED_REPO / "comparison_outputs" / "optimized"
SIDE_BY_SIDE_OUT = OPTIMIZED_REPO / "comparison_outputs" / "side_by_side"

# Images with clearly visible hands
IMAGES = [
    OPTIMIZED_REPO / "example_data" / "images" / "pexels-anete-lusina-4793258.jpg",
    OPTIMIZED_REPO / "test_images_new" / "pushup.jpg",
    OPTIMIZED_REPO / "test_images_new" / "reaching.jpg",
    OPTIMIZED_REPO / "test_images_new" / "climbing.jpg",
]


def prepare_inputs():
    INPUT_DIR.mkdir(parents=True, exist_ok=True)
    for src in IMAGES:
        dst = INPUT_DIR / src.name
        if not dst.exists():
            shutil.copy(src, dst)
            print(f"Copied {src.name} -> {dst}")
        else:
            print(f"Already exists: {dst}")


def run_baseline():
    BASE_OUT.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable,
        str(BASELINE_REPO / "demo.py"),
        "--img_folder", str(INPUT_DIR),
        "--out_folder", str(BASE_OUT),
        "--batch_size", "8",
    ]
    print("\n[BASELINE] Running:", " ".join(cmd))
    env = os.environ.copy()
    env["PYOPENGL_PLATFORM"] = "osmesa"
    env["MESA_GL_VERSION_OVERRIDE"] = "3.3"
    env["MESA_GLSL_VERSION_OVERRIDE"] = "330"
    env["LD_LIBRARY_PATH"] = OSMESA_LIB + os.pathsep + env.get("LD_LIBRARY_PATH", "")
    subprocess.run(cmd, cwd=BASELINE_REPO, env=env, check=True)


def run_optimized():
    OPT_OUT.mkdir(parents=True, exist_ok=True)
    for img_path in sorted(INPUT_DIR.glob("*")):
        if img_path.suffix.lower() not in {".jpg", ".jpeg", ".png"}:
            continue
        out_dir = OPT_OUT / img_path.stem
        out_dir.mkdir(parents=True, exist_ok=True)
        cmd = [
            sys.executable,
            str(OPTIMIZED_REPO / "demo.py"),
            "--img", str(img_path),
            "--smplx", SMPLX_PATH,
            "--out", str(out_dir),
            "--device", "cuda",
        ]
        print(f"\n[OPTIMIZED] Running on {img_path.name}:", " ".join(cmd))
        env = os.environ.copy()
        env["PYOPENGL_PLATFORM"] = "osmesa"
        env["MESA_GL_VERSION_OVERRIDE"] = "3.3"
        env["MESA_GLSL_VERSION_OVERRIDE"] = "330"
        env["LD_LIBRARY_PATH"] = OSMESA_LIB + os.pathsep + env.get("LD_LIBRARY_PATH", "")
        env["HAMER_CACHE_DIR"] = HAMER_CACHE_DIR
        subprocess.run(cmd, cwd=OPTIMIZED_REPO, env=env, check=True)


def add_label(img, label, color=(0, 0, 0), bg_color=(255, 255, 255)):
    """Add a text label at the top of an image."""
    h, w = img.shape[:2]
    label_h = 40
    canvas = np.ones((h + label_h, w, 3), dtype=np.uint8) * np.array(bg_color, dtype=np.uint8)
    canvas[label_h:, :] = img
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 1.0
    thickness = 2
    text_size = cv2.getTextSize(label, font, scale, thickness)[0]
    x = (w - text_size[0]) // 2
    y = int(label_h * 0.75)
    cv2.putText(canvas, label, (x, y), font, scale, color, thickness, cv2.LINE_AA)
    return canvas


def make_side_by_side():
    SIDE_BY_SIDE_OUT.mkdir(parents=True, exist_ok=True)

    for img_path in sorted(INPUT_DIR.glob("*")):
        if img_path.suffix.lower() not in {".jpg", ".jpeg", ".png"}:
            continue
        stem = img_path.stem

        # Baseline outputs: {stem}_0.png, {stem}_1.png, ...
        baseline_imgs = sorted(BASE_OUT.glob(f"{stem}_*.png"))
        # Optimized outputs in subfolder: person_{i}_detail.png
        opt_dir = OPT_OUT / stem
        opt_imgs = sorted(opt_dir.glob("person_*_detail.png"))

        if not baseline_imgs or not opt_imgs:
            print(f"Skipping {stem}: missing outputs (baseline={len(baseline_imgs)}, opt={len(opt_imgs)})")
            continue

        n = min(len(baseline_imgs), len(opt_imgs))
        for i in range(n):
            b_img = cv2.imread(str(baseline_imgs[i]))
            o_img = cv2.imread(str(opt_imgs[i]))
            if b_img is None or o_img is None:
                continue

            # Resize to same height
            target_h = min(b_img.shape[0], o_img.shape[0], 1080)
            if b_img.shape[0] != target_h:
                b_img = cv2.resize(b_img, (int(b_img.shape[1] * target_h / b_img.shape[0]), target_h))
            if o_img.shape[0] != target_h:
                o_img = cv2.resize(o_img, (int(o_img.shape[1] * target_h / o_img.shape[0]), target_h))

            b_img = add_label(b_img, "Baseline (4D-Humans)", color=(255, 255, 255), bg_color=(180, 80, 60))
            o_img = add_label(o_img, "Optimized (Body + Hand)", color=(255, 255, 255), bg_color=(60, 120, 180))

            combined = np.concatenate([b_img, o_img], axis=1)
            out_path = SIDE_BY_SIDE_OUT / f"{stem}_person{i}_compare.png"
            cv2.imwrite(str(out_path), combined)
            print(f"Saved comparison: {out_path}")

        # Also make a summary strip for this image if multiple people
        if n > 1:
            summary_imgs = [cv2.imread(str(SIDE_BY_SIDE_OUT / f"{stem}_person{i}_compare.png")) for i in range(n)]
            summary_imgs = [im for im in summary_imgs if im is not None]
            if summary_imgs:
                # resize to same width
                target_w = min(im.shape[1] for im in summary_imgs)
                summary_imgs = [cv2.resize(im, (target_w, int(im.shape[0] * target_w / im.shape[1]))) for im in summary_imgs]
                summary = np.concatenate(summary_imgs, axis=0)
                out_path = SIDE_BY_SIDE_OUT / f"{stem}_all_compare.png"
                cv2.imwrite(str(out_path), summary)
                print(f"Saved summary: {out_path}")


def main():
    prepare_inputs()
    run_baseline()
    run_optimized()
    make_side_by_side()
    print("\nAll done. Comparisons saved to:", SIDE_BY_SIDE_OUT)


if __name__ == "__main__":
    main()
