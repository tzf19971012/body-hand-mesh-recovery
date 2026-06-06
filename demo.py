#!/usr/bin/env python3
"""
Self-contained combined body + hand inference using HMR2 + HaMeR + SMPL-X.
Saves SMPL-X mesh with body from HMR2 and hands from HaMeR.

Setup:
    1. pip install -r requirements.txt
    2. Download SMPL-X neutral model (SMPLX_NEUTRAL_py3.pkl) and set --smplx.
    3. Run fetch scripts to download HMR2 / HaMeR / ViTPose checkpoints.

Example:
    export SMPLX_PATH=/path/to/SMPLX_NEUTRAL_py3.pkl
    python demo.py --img /path/to/image.jpg --out output
"""

import os
import sys
import argparse

# osmesa must be set BEFORE pyrender is imported
os.environ['PYOPENGL_PLATFORM'] = 'osmesa'
os.environ['MESA_GL_VERSION_OVERRIDE'] = '3.3'
os.environ['MESA_GLSL_VERSION_OVERRIDE'] = '330'

import torch
import numpy as np
import cv2
from pathlib import Path

# ------------------------------------------------------------------
# Local packages (copied from 4D-Humans and HaMeR repos)
# ------------------------------------------------------------------
from hmr2.models import load_hmr2, DEFAULT_CHECKPOINT as BODY_CKPT
from hmr2.utils import recursive_to
from hmr2.datasets.vitdet_dataset import ViTDetDataset as BodyDataset
from hmr2.utils.renderer import cam_crop_to_full

import hamer.configs as _hamer_cfg
_hamer_cfg.CACHE_DIR_HAMER = os.environ.get('HAMER_CACHE_DIR', './_DATA')

from hamer.models import load_hamer, DEFAULT_CHECKPOINT as HAND_CKPT
from hamer.datasets.vitdet_dataset import ViTDetDataset as HandDataset

# vitpose_model uses relative ROOT_DIR = "./"; fix it to absolute project root
import vitpose_model
_PROJECT_ROOT = str(Path(__file__).parent.resolve())
# Use the env var directly (bypassing any cached relative path)
_CACHE_DIR = str(Path(os.environ.get('HAMER_CACHE_DIR', './_DATA')).resolve())
vitpose_model.ROOT_DIR = _PROJECT_ROOT
vitpose_model.VIT_DIR = os.path.join(_PROJECT_ROOT, "third_party", "ViTPose")
for name, dic in vitpose_model.ViTPoseModel.MODEL_DICT.items():
    cfg_rel = dic['config'].replace('./', '').replace('third-party', 'third_party').lstrip('/')
    dic['config'] = os.path.join(_PROJECT_ROOT, cfg_rel)
    # Checkpoint lives inside the HaMeR cache dir
    # Original model path:  ./{ROOT_DIR}/_DATA/...
    # We want:              {CACHE_DIR}/vitpose_ckpts/...
    mdl_rel = dic['model'].replace('./', '').lstrip('/')
    # Strip the leading _DATA/ because _CACHE_DIR already points to it
    if mdl_rel.startswith('_DATA/'):
        mdl_rel = mdl_rel[len('_DATA/'):]
    dic['model'] = os.path.join(_CACHE_DIR, mdl_rel)

os.environ['PYOPENGL_PLATFORM'] = 'osmesa'

import smplx


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------

def load_models(device, smplx_path):
    print("[1/4] Loading HMR2 (body)...")
    body_model, body_cfg = load_hmr2(BODY_CKPT)
    body_model = body_model.to(device).eval()

    print("[2/4] Loading HaMeR (hand)...")
    hand_model, hand_cfg = load_hamer(HAND_CKPT)
    hand_model = hand_model.to(device).eval()

    print("[3/4] Loading SMPL-X...")
    smplx_model = smplx.create(
        model_path=smplx_path,
        model_type='smplx',
        gender='neutral',
        num_betas=10,
        use_pca=False,
        flat_hand_mean=True,
        batch_size=1
    ).to(device).eval()

    return body_model, body_cfg, hand_model, hand_cfg, smplx_model


def detect_bodies(img_cv2, device):
    from hmr2.utils.utils_detectron2 import DefaultPredictor_Lazy
    from detectron2.config import LazyConfig
    import hmr2

    cfg_path = Path(hmr2.__file__).parent / 'configs' / 'cascade_mask_rcnn_vitdet_h_75ep.py'
    detectron2_cfg = LazyConfig.load(str(cfg_path))
    detectron2_cfg.train.init_checkpoint = (
        "https://dl.fbaipublicfiles.com/detectron2/ViTDet/COCO/cascade_mask_rcnn_vitdet_h/f328730692/model_final_f05665.pkl"
    )
    for i in range(3):
        detectron2_cfg.model.roi_heads.box_predictors[i].test_score_thresh = 0.25
    detector = DefaultPredictor_Lazy(detectron2_cfg)

    det_out = detector(img_cv2)
    det_instances = det_out['instances']
    valid_idx = (det_instances.pred_classes == 0) & (det_instances.scores > 0.5)
    boxes = det_instances.pred_boxes.tensor[valid_idx].cpu().numpy()
    return boxes


def detect_hands(img_cv2, boxes, device):
    cpm = vitpose_model.ViTPoseModel(device)
    img = img_cv2[:, :, ::-1].copy()
    scores = np.ones((len(boxes), 1))
    vitposes_out = cpm.predict_pose(img, [np.concatenate([boxes, scores], axis=1)])

    hand_bboxes, hand_is_right, hand_person_ids = [], [], []
    for person_id, vitposes in enumerate(vitposes_out):
        left_hand_keyp = vitposes['keypoints'][-42:-21]
        right_hand_keyp = vitposes['keypoints'][-21:]
        for keyp, is_right in [(left_hand_keyp, 0), (right_hand_keyp, 1)]:
            valid = keyp[:, 2] > 0.5
            if valid.sum() > 3:
                bbox = [keyp[valid, 0].min(), keyp[valid, 1].min(),
                        keyp[valid, 0].max(), keyp[valid, 1].max()]
                hand_bboxes.append(bbox)
                hand_is_right.append(is_right)
                hand_person_ids.append(person_id)

    if len(hand_bboxes) == 0:
        return None, None, None
    return np.stack(hand_bboxes), np.stack(hand_is_right), np.stack(hand_person_ids)


def infer_body(body_model, body_cfg, img_cv2, boxes, device):
    dataset = BodyDataset(body_cfg, img_cv2, boxes)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=8, shuffle=False, num_workers=0)
    results = []
    for batch in dataloader:
        batch = recursive_to(batch, device)
        with torch.no_grad():
            out = body_model(batch)
        B = batch['img'].shape[0]
        for i in range(B):
            results.append({
                'pred_smpl_params': {k: v[i:i+1] for k, v in out['pred_smpl_params'].items()},
                'pred_cam': out['pred_cam'][i:i+1],
                'pred_vertices': out['pred_vertices'][i:i+1],
                'pred_cam_t': out['pred_cam_t'][i:i+1],
                'box_center': batch['box_center'][i],
                'box_size': batch['box_size'][i],
                'img_size': batch['img_size'][i],
            })
    return results


def infer_hands(hand_model, hand_cfg, img_cv2, hand_boxes, hand_is_right, hand_person_ids, device):
    dataset = HandDataset(hand_cfg, img_cv2, hand_boxes, hand_is_right, rescale_factor=2.0)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=8, shuffle=False, num_workers=0)
    results = []
    idx = 0
    for batch in dataloader:
        batch = recursive_to(batch, device)
        with torch.no_grad():
            out = hand_model(batch)
        B = batch['img'].shape[0]

        multiplier = (2 * batch['right'] - 1)
        pred_cam = out['pred_cam'].clone()
        pred_cam[:, 1] = multiplier * pred_cam[:, 1]
        box_center = batch['box_center'].float()
        box_size = batch['box_size'].float()
        img_size = batch['img_size'].float()
        scaled_focal_length = hand_cfg.EXTRA.FOCAL_LENGTH / hand_cfg.MODEL.IMAGE_SIZE * img_size.max()
        pred_cam_t_full = cam_crop_to_full(pred_cam, box_center, box_size, img_size, scaled_focal_length)

        for i in range(B):
            verts = out['pred_vertices'][i].detach().cpu().numpy()
            is_right = int(batch['right'][i].item())
            verts[:, 0] = (2 * is_right - 1) * verts[:, 0]
            cam_t = pred_cam_t_full[i].detach().cpu().numpy()

            results.append({
                'pred_mano_params': {k: v[i:i+1] for k, v in out['pred_mano_params'].items()},
                'pred_vertices': verts,
                'pred_cam_t': cam_t,
                'right': is_right,
                'person_id': int(hand_person_ids[idx]),
            })
            idx += 1
    return results


# ------------------------------------------------------------------
# SMPL kinematic-tree helpers for wrist rotation transfer
# ------------------------------------------------------------------

SMPL_PARENTS = [
    -1, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 9, 9, 12, 13, 14, 16, 17, 18, 19
]

LEFT_WRIST_CHAIN = [2, 5, 8, 12, 15, 17]
RIGHT_WRIST_CHAIN = [2, 5, 8, 13, 16, 18]


def wrist_chain_rotation(global_orient, body_pose, chain):
    R = global_orient.clone()
    if R.dim() == 3 and R.shape[0] == 1:
        R = R.squeeze(0)
    for idx in chain:
        joint_rot = body_pose[idx]
        if joint_rot.dim() == 3 and joint_rot.shape[0] == 1:
            joint_rot = joint_rot.squeeze(0)
        R = R @ joint_rot
    return R


def rotmat_to_aa(rotmat):
    if rotmat.dim() == 2:
        rotmat = rotmat.unsqueeze(0)
        squeeze = True
    else:
        squeeze = False

    batch_size = rotmat.shape[0]
    trace = rotmat[:, 0, 0] + rotmat[:, 1, 1] + rotmat[:, 2, 2]
    angle = torch.acos(torch.clamp((trace - 1) / 2, -1.0, 1.0))

    axis = torch.stack([
        rotmat[:, 2, 1] - rotmat[:, 1, 2],
        rotmat[:, 0, 2] - rotmat[:, 2, 0],
        rotmat[:, 1, 0] - rotmat[:, 0, 1]
    ], dim=1)

    axis_norm = torch.norm(axis, dim=1, keepdim=True) + 1e-8
    axis = axis / axis_norm

    mask_small = angle < 1e-6
    axis[mask_small] = 0

    aa = axis * angle.unsqueeze(1)

    mask_pi = angle > 3.14159 - 1e-4
    if mask_pi.any():
        diag = torch.stack([
            rotmat[:, 0, 0], rotmat[:, 1, 1], rotmat[:, 2, 2]
        ], dim=1)
        axis_sq = torch.clamp((diag + 1) / 2, 0, 1)
        axis_sign = torch.sign(axis)
        axis_pi = torch.sqrt(axis_sq) * axis_sign
        axis_pi = axis_pi / (torch.norm(axis_pi, dim=1, keepdim=True) + 1e-8)
        aa[mask_pi] = axis_pi[mask_pi] * angle[mask_pi].unsqueeze(1)

    if squeeze:
        aa = aa.squeeze(0)
    return aa


def combine_smplx(body_results, hand_results, smplx_model, body_cfg):
    device = next(smplx_model.parameters()).device
    n_persons = len(body_results)
    if n_persons == 0:
        return None, None

    left_hand_pose = torch.zeros(n_persons, 15, 3, 3, device=device)
    right_hand_pose = torch.zeros(n_persons, 15, 3, 3, device=device)

    global_orient = torch.cat([b['pred_smpl_params']['global_orient'] for b in body_results], dim=0)
    body_pose = torch.cat([b['pred_smpl_params']['body_pose'] for b in body_results], dim=0)
    betas = torch.cat([b['pred_smpl_params']['betas'] for b in body_results], dim=0)

    body_pose = body_pose[:, :21].clone()

    for h in hand_results:
        pid = h['person_id']
        if pid >= n_persons:
            continue

        hand_pose = h['pred_mano_params']['hand_pose']
        mano_global = h['pred_mano_params']['global_orient']
        R_mano = mano_global[0]
        if R_mano.dim() == 3 and R_mano.shape[0] == 1:
            R_mano = R_mano.squeeze(0)

        if h['right'] == 1:
            right_hand_pose[pid] = hand_pose[0]
            R_target = R_mano
            R_chain = wrist_chain_rotation(
                global_orient[pid], body_pose[pid], RIGHT_WRIST_CHAIN
            )
            body_pose[pid, 20] = R_chain.mT @ R_target
        else:
            Rz180 = torch.tensor([[-1., 0., 0.],
                                  [0., -1., 0.],
                                  [0., 0., 1.]],
                                 device=device, dtype=torch.float32)
            left_hand_pose[pid] = hand_pose[0] @ Rz180
            R_target = R_mano @ Rz180
            R_chain = wrist_chain_rotation(
                global_orient[pid], body_pose[pid], LEFT_WRIST_CHAIN
            )
            body_pose[pid, 19] = R_chain.mT @ R_target

    global_orient_aa = rotmat_to_aa(global_orient.reshape(-1, 3, 3))
    body_pose_aa = rotmat_to_aa(body_pose.reshape(-1, 3, 3)).reshape(n_persons, 21, 3)
    left_hand_pose_aa = rotmat_to_aa(left_hand_pose.reshape(-1, 3, 3)).reshape(n_persons, 15, 3)
    right_hand_pose_aa = rotmat_to_aa(right_hand_pose.reshape(-1, 3, 3)).reshape(n_persons, 15, 3)

    smplx_out = smplx_model(
        global_orient=global_orient_aa.float(),
        body_pose=body_pose_aa.float(),
        left_hand_pose=left_hand_pose_aa.float(),
        right_hand_pose=right_hand_pose_aa.float(),
        betas=betas.float(),
        jaw_pose=torch.zeros(n_persons, 3, device=device),
        leye_pose=torch.zeros(n_persons, 3, device=device),
        reye_pose=torch.zeros(n_persons, 3, device=device),
        expression=torch.zeros(n_persons, 10, device=device),
    )

    cam_t_full = []
    for i in range(n_persons):
        b = body_results[i]
        scaled_focal_length = body_cfg.EXTRA.FOCAL_LENGTH / body_cfg.MODEL.IMAGE_SIZE * b['img_size'].max()
        cam_t_full.append(cam_crop_to_full(
            b['pred_cam'],
            b['box_center'].unsqueeze(0),
            b['box_size'].unsqueeze(0),
            b['img_size'].unsqueeze(0),
            scaled_focal_length
        )[0])

    return smplx_out.vertices, cam_t_full


def save_obj(vertices, faces, path):
    with open(path, 'w') as f:
        for v in vertices:
            f.write(f"v {v[0]:.6f} {v[1]:.6f} {v[2]:.6f}\n")
        for face in faces:
            f.write(f"f {face[0]+1} {face[1]+1} {face[2]+1}\n")


# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Body + Hand inference (HMR2 + HaMeR + SMPL-X)")
    parser.add_argument("--img", required=True, help="Input image path")
    parser.add_argument("--smplx", default=os.environ.get('SMPLX_PATH'), help="Path to SMPLX_NEUTRAL_py3.pkl")
    parser.add_argument("--out", default="output", help="Output directory")
    parser.add_argument("--device", default="cuda", help="Device: cuda or cpu")
    args = parser.parse_args()

    if not args.smplx:
        raise ValueError("Please provide --smplx or set SMPLX_PATH environment variable.")

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    out_dir = args.out
    os.makedirs(out_dir, exist_ok=True)

    body_model, body_cfg, hand_model, hand_cfg, smplx_model = load_models(device, args.smplx)

    img_cv2 = cv2.imread(str(args.img))
    if img_cv2 is None:
        raise FileNotFoundError(f"Image not found: {args.img}")

    print("[4/4] Detecting & inferring...")
    boxes = detect_bodies(img_cv2, device)
    if len(boxes) == 0:
        print("No bodies detected.")
        return
    print(f"  -> {len(boxes)} body(s)")

    hand_boxes, hand_is_right, hand_person_ids = detect_hands(img_cv2, boxes, device)
    n_hands = len(hand_boxes) if hand_boxes is not None else 0
    print(f"  -> {n_hands} hand(s)")

    body_results = infer_body(body_model, body_cfg, img_cv2, boxes, device)
    hand_results = []
    if hand_boxes is not None:
        hand_results = infer_hands(hand_model, hand_cfg, img_cv2, hand_boxes, hand_is_right, hand_person_ids, device)

    vertices, cam_t_list = combine_smplx(body_results, hand_results, smplx_model, body_cfg)
    if vertices is None:
        print("No output.")
        return

    faces = smplx_model.faces
    for i in range(len(body_results)):
        verts = vertices[i].detach().cpu().numpy() + cam_t_list[i].detach().cpu().numpy()
        out_path = os.path.join(out_dir, f'person_{i}_smplx.obj')
        save_obj(verts, faces, out_path)
        print(f"  Saved: {out_path}")

    print("[5/5] Rendering...")
    from hmr2.utils.renderer import Renderer
    renderer = Renderer(body_cfg, faces=faces)
    for i in range(len(body_results)):
        rendered = renderer(
            vertices=vertices[i].detach().cpu().numpy(),
            camera_translation=cam_t_list[i].detach().cpu().numpy(),
            image=None,
            full_frame=True,
            imgname=str(args.img),
        )
        rend_path = os.path.join(out_dir, f'person_{i}_rendered.png')
        cv2.imwrite(rend_path, (rendered[:, :, ::-1] * 255).astype(np.uint8))
        print(f"  Rendered: {rend_path}")

    print("Done.")


if __name__ == '__main__':
    main()
