#!/usr/bin/env python3
"""
colmap_camera_generator.py

Generate a camera.json file (COLMAP-like JSON entries) describing a camera path
that orbits around a target point on a circular trajectory.

Features:
- Orbit around a specified axis (x, y, or z).
- Choose radius, number of frames, and center point.
- Output rotation as either world->camera (R_w2c) or camera->world (R_c2w).
- Output translation as camera center (C) or as the COLMAP-style translation t = -R * C.
- Set intrinsics (fx, fy, cx, cy, width, height), image name prefix and start index.

The JSON format produced matches the structure shown in your example — a list of
entries, each with keys: camera_id, model, image_name, fx, fy, cx, cy, width, height,
rotation (3x3 list), translation (3 list).

If your importer strictly expects a different convention, use the --rotation_conv and
--translation_mode flags to switch conventions.

Example:
python3 colmap_camera_generator.py \
  --axis y --radius 3.8 --num-frames 36 --center 0 0 0 \
  --output camera.json --image_prefix templeR --start_index 1 \
  --fx 1552.5417644224121 --fy 1552.5417644224121 --cx 320 --cy 240 \
  --width 640 --height 480

"""

import argparse
import json
import math
from typing import List, Tuple

import numpy as np


def normalize(v: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(v)
    if n < 1e-12:
        return v
    return v / n


def look_at_rotation(camera_pos: np.ndarray, target: np.ndarray, up_hint: np.ndarray = None) -> Tuple[np.ndarray, np.ndarray]:
    """
    Build camera rotation matrices from camera position and target "look-at" point.

    Returns (R_c2w, R_w2c)
    - R_c2w: 3x3 rotation matrix mapping camera coordinates -> world coordinates
             (columns are the camera axes in world frame: right, up, forward)
    - R_w2c: its transpose (world -> camera)

    By convention here forward = normalize(target - camera_pos) (i.e. camera looks towards
    the target from the camera position).
    """
    if up_hint is None:
        up_hint = np.array([0.0, 1.0, 0.0])

    forward = normalize(np.array(target) - np.array(camera_pos))
    # If forward is collinear with up_hint, choose a different up
    up_hint = normalize(np.array(up_hint))
    if abs(np.dot(forward, up_hint)) > 0.999:
        up_hint = np.array([1.0, 0.0, 0.0])

    right = normalize(np.cross(forward, up_hint))
    up = np.cross(right, forward)

    # R_c2w: columns are [right, up, forward]
    R_c2w = np.stack([right, up, forward], axis=1)
    R_w2c = R_c2w.T
    return R_c2w, R_w2c


def circle_positions(axis: str, radius: float, center: np.ndarray, num_frames: int, start_angle_deg: float = 0.0) -> List[np.ndarray]:
    """
    Generate camera center positions on a circle around `center` on the plane perpendicular
    to `axis`.

    - axis: 'x', 'y', or 'z' (axis to rotate around)
    - radius: radius of circular path
    - center: 3-array center of rotation
    - num_frames: how many discrete positions
    - start_angle_deg: starting angle offset (degrees)
    """
    centers = []
    for i in range(num_frames):
        theta = 2.0 * math.pi * (i / float(num_frames)) + math.radians(start_angle_deg)
        if axis == 'y':
            # rotate around y: circle in x-z plane
            x = center[0] + radius * math.cos(theta)
            y = center[1]
            z = center[2] + radius * math.sin(theta)
        elif axis == 'x':
            # rotate around x: circle in y-z plane
            x = center[0]
            y = center[1] + radius * math.cos(theta)
            z = center[2] + radius * math.sin(theta)
        elif axis == 'z':
            # rotate around z: circle in x-y plane
            x = center[0] + radius * math.cos(theta)
            y = center[1] + radius * math.sin(theta)
            z = center[2]
        else:
            raise ValueError("axis must be 'x', 'y' or 'z'")
        centers.append(np.array([x, y, z], dtype=float))
    return centers


def matrix_to_list3x3(m: np.ndarray) -> List[List[float]]:
    return [[float(m[r, c]) for c in range(3)] for r in range(3)]


def vector_to_list3(v: np.ndarray) -> List[float]:
    return [float(v[i]) for i in range(3)]


def generate_camera_entries(
    axis: str,
    radius: float,
    num_frames: int,
    center: Tuple[float, float, float],
    up: Tuple[float, float, float],
    start_angle_deg: float,
    image_prefix: str,
    start_index: int,
    intrinsics: dict,
    camera_id: int,
    model: str,
    rotation_conv: str,
    translation_mode: str,
    pad: int = 4,
):
    centers = circle_positions(axis, radius, np.array(center, dtype=float), num_frames, start_angle_deg)

    entries = []
    for i, cam_pos in enumerate(centers):
        # build rotation matrices
        R_c2w, R_w2c = look_at_rotation(cam_pos, np.array(center, dtype=float), up_hint=np.array(up, dtype=float))

        # choose rotation matrix convention
        if rotation_conv == 'w2c':
            R_out = R_w2c
        elif rotation_conv == 'c2w':
            R_out = R_c2w
        else:
            raise ValueError("rotation_conv must be 'w2c' or 'c2w'")

        # choose translation mode
        if translation_mode == 'center':
            t_out = cam_pos
        elif translation_mode == 't':
            # t = -R_w2c * C  (world->camera translation)
            t_out = -R_w2c.dot(cam_pos)
        else:
            raise ValueError("translation_mode must be 'center' or 't'")

        img_idx = start_index + i
        img_name = f"{image_prefix}{str(img_idx).zfill(pad)}.png"

        entry = {
            'camera_id': camera_id,
            'model': model,
            'image_name': img_name,
            'fx': float(intrinsics['fx']),
            'fy': float(intrinsics['fy']),
            'cx': float(intrinsics['cx']),
            'cy': float(intrinsics['cy']),
            'width': float(intrinsics['width']),
            'height': float(intrinsics['height']),
            'rotation': matrix_to_list3x3(R_out),
            'translation': vector_to_list3(t_out),
        }
        entries.append(entry)
    return entries


def cli():
    p = argparse.ArgumentParser(description='Generate camera.json with orbiting cameras (COLMAP-like format).')
    p.add_argument('--axis', choices=['x', 'y', 'z'], default='y', help='Axis to orbit around (default: y)')
    p.add_argument('--radius', type=float, required=True, help='Radius of the orbit')
    p.add_argument('--num-frames', type=int, default=36, help='Number of frames to generate')
    p.add_argument('--center', type=float, nargs=3, default=[0.0, 0.0, 0.0], help='Center of orbit (x y z)')
    p.add_argument('--up', type=float, nargs=3, default=[0.0, 1.0, 0.0], help='Up vector hint for look-at')
    p.add_argument('--start-angle-deg', type=float, default=0.0, help='Starting angle in degrees')
    p.add_argument('--image-prefix', type=str, default='image', help='Image filename prefix')
    p.add_argument('--start-index', type=int, default=1, help='Start index for image filenames')
    p.add_argument('--pad', type=int, default=4, help='Zero-pad width for image index (e.g. 4 -> 0001)')
    p.add_argument('--fx', type=float, required=True)
    p.add_argument('--fy', type=float, required=True)
    p.add_argument('--cx', type=float, required=True)
    p.add_argument('--cy', type=float, required=True)
    p.add_argument('--width', type=float, required=True)
    p.add_argument('--height', type=float, required=True)
    p.add_argument('--camera-id', type=int, default=1)
    p.add_argument('--model', type=str, default='SIMPLE_RADIAL')
    p.add_argument('--rotation-conv', choices=['w2c', 'c2w'], default='w2c',
                   help="Rotation output convention: 'w2c' (world->camera) or 'c2w' (camera->world).")
    p.add_argument('--translation-mode', choices=['center', 't'], default='center',
                   help="Translation output: 'center' to write camera center coords, or 't' to write t = -R * C (world->camera translation).")
    p.add_argument('--output', type=str, default='camera.json', help='Output JSON filename')

    args = p.parse_args()

    intr = {'fx': args.fx, 'fy': args.fy, 'cx': args.cx, 'cy': args.cy, 'width': args.width, 'height': args.height}

    entries = generate_camera_entries(
        axis=args.axis,
        radius=args.radius,
        num_frames=args.num_frames,
        center=tuple(args.center),
        up=tuple(args.up),
        start_angle_deg=args.start_angle_deg,
        image_prefix=args.image_prefix,
        start_index=args.start_index,
        intrinsics=intr,
        camera_id=args.camera_id,
        model=args.model,
        rotation_conv=args.rotation_conv,
        translation_mode=args.translation_mode,
        pad=args.pad,
    )

    with open(args.output, 'w') as f:
        json.dump(entries, f, indent=4)

    print(f'Wrote {len(entries)} camera entries to {args.output}')


if __name__ == '__main__':
    cli()
