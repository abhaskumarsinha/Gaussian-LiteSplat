#!/usr/bin/env python3
"""
colmap_camera_generator.py


New features summary (no need to open the file to read this — the canvas contains the code):
- --bbox minx miny minz maxx maxy maxz : compute center and automatic radius from object bounds
- --object-centers-file FILE : load JSON array of object centers [[x,y,z], ...]
- --radius-scale SCALE : multiply auto-computed radius by this factor
- --orbit-each : if provided, the script will generate an orbit around *each* object center
                  sequentially (use --frames-per-object to control frames per object)
- --frames-per-object N : used with --orbit-each to specify how many frames around each object
- If neither --bbox nor --object-centers-file are provided, behavior falls back to single
  center at --center (same as earlier script).

Examples:
1) Orbit around a bounding box center, auto radius:
python3 colmap_camera_generator.py --bbox -1 -1 -1 1 2 3 --radius-scale 1.2 --num-frames 60 \
  --fx 1552.54 --fy 1552.54 --cx 320 --cy 240 --width 640 --height 480 \
  --image-prefix templeR --output camera.json

2) Orbit each object center (three objects) with 30 frames each:
python3 colmap_camera_generator.py --object-centers-file centers.json --orbit-each --frames-per-object 30 \
  --fx 1552.54 --fy 1552.54 --cx 320 --cy 240 --width 640 --height 480 --image-prefix templeR

"""

import argparse
import json
import math
import os
from typing import List, Tuple

import numpy as np


def normalize(v: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(v)
    if n < 1e-12:
        return v
    return v / n


def look_at_rotation(camera_pos: np.ndarray, target: np.ndarray, up_hint: np.ndarray = None) -> Tuple[np.ndarray, np.ndarray]:
    if up_hint is None:
        up_hint = np.array([0.0, 1.0, 0.0])

    forward = normalize(np.array(target) - np.array(camera_pos))
    up_hint = normalize(np.array(up_hint))
    if abs(np.dot(forward, up_hint)) > 0.999:
        up_hint = np.array([1.0, 0.0, 0.0])

    right = normalize(np.cross(forward, up_hint))
    up = np.cross(right, forward)

    R_c2w = np.stack([right, up, forward], axis=1)
    R_w2c = R_c2w.T
    return R_c2w, R_w2c


def circle_positions(axis: str, radius: float, center: np.ndarray, num_frames: int, start_angle_deg: float = 0.0) -> List[np.ndarray]:
    centers = []
    for i in range(num_frames):
        theta = 2.0 * math.pi * (i / float(num_frames)) + math.radians(start_angle_deg)
        if axis == 'y':
            x = center[0] + radius * math.cos(theta)
            y = center[1]
            z = center[2] + radius * math.sin(theta)
        elif axis == 'x':
            x = center[0]
            y = center[1] + radius * math.cos(theta)
            z = center[2] + radius * math.sin(theta)
        elif axis == 'z':
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


def compute_center_and_radius_from_bbox(bbox: List[float], radius_scale: float = 1.0) -> Tuple[np.ndarray, float]:
    minp = np.array(bbox[:3], dtype=float)
    maxp = np.array(bbox[3:], dtype=float)
    center = 0.5 * (minp + maxp)
    diag = np.linalg.norm(maxp - minp)
    radius = 0.5 * diag * radius_scale
    return center, float(radius)


def load_object_centers_from_file(path: str) -> List[np.ndarray]:
    with open(path, 'r') as f:
        data = json.load(f)
    centers = [np.array(c, dtype=float) for c in data]
    return centers


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
        R_c2w, R_w2c = look_at_rotation(cam_pos, np.array(center, dtype=float), up_hint=np.array(up, dtype=float))

        if rotation_conv == 'w2c':
            R_out = R_w2c
        else:
            R_out = R_c2w

        if translation_mode == 'center':
            t_out = cam_pos
        else:
            t_out = -R_w2c.dot(cam_pos)

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
    p.add_argument('--radius', type=float, default=None, help='Radius of the orbit (if omitted, computed from bbox or object centers)')
    p.add_argument('--num-frames', type=int, default=36, help='Number of frames to generate (used when not orbiting each object)')
    p.add_argument('--frames-per-object', type=int, default=36, help='Frames per object when using --orbit-each')
    p.add_argument('--center', type=float, nargs=3, default=[0.0, 0.0, 0.0], help='Fallback center of orbit (x y z)')
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
    p.add_argument('--bbox', type=float, nargs=6, default=None,
                   help='Bounding box as minx miny minz maxx maxy maxz (used to compute center and auto radius)')
    p.add_argument('--object-centers-file', type=str, default=None,
                   help='JSON file containing array of object centers [[x,y,z], ...]')
    p.add_argument('--radius-scale', type=float, default=1.0, help='Scale applied to auto-computed radius from bbox')
    p.add_argument('--orbit-each', action='store_true', help='If set, orbit around each object center sequentially')
    p.add_argument('--output', type=str, default='camera.json', help='Output JSON filename')

    args = p.parse_args()

    intr = {'fx': args.fx, 'fy': args.fy, 'cx': args.cx, 'cy': args.cy, 'width': args.width, 'height': args.height}

    # Determine object centers and radius
    object_centers = None
    bbox_center = None
    auto_radius = None

    if args.object_centers_file:
        if not os.path.exists(args.object_centers_file):
            raise FileNotFoundError(f"Object centers file not found: {args.object_centers_file}")
        object_centers = load_object_centers_from_file(args.object_centers_file)

    if args.bbox is not None:
        bbox_center, auto_radius = compute_center_and_radius_from_bbox(args.bbox, radius_scale=args.radius_scale)

    # If object centers provided and orbit-each -> orbit each center separately
    entries = []
    cur_index = args.start_index

    if args.orbit_each and object_centers is not None:
        for obj_idx, obj_center in enumerate(object_centers):
            r = args.radius if args.radius is not None else (auto_radius if auto_radius is not None else np.linalg.norm(obj_center - np.array(args.center)))
            frames = args.frames_per_object
            img_prefix = f"{args.image_prefix}_obj{obj_idx}_"
            gen = generate_camera_entries(
                axis=args.axis,
                radius=float(r),
                num_frames=frames,
                center=tuple(obj_center.tolist()),
                up=tuple(args.up),
                start_angle_deg=args.start_angle_deg,
                image_prefix=img_prefix,
                start_index=cur_index,
                intrinsics=intr,
                camera_id=args.camera_id,
                model=args.model,
                rotation_conv=args.rotation_conv,
                translation_mode=args.translation_mode,
                pad=args.pad,
            )
            entries.extend(gen)
            cur_index += frames

    else:
        # compute a single center: prefer bbox center, then centroid of object_centers, then fallback to args.center
        if bbox_center is not None:
            center = bbox_center
        elif object_centers is not None:
            center = np.mean(np.stack([c for c in object_centers], axis=0), axis=0)
        else:
            center = np.array(args.center, dtype=float)

        # compute radius: prefer user radius, then auto_radius, then distance of first object center (if any), otherwise user radius or default
        if args.radius is not None:
            r = float(args.radius)
        elif auto_radius is not None:
            r = auto_radius
        elif object_centers is not None and len(object_centers) > 0:
            # set radius to average distance from centroid to object centers (if multiple)
            dists = [np.linalg.norm(c - center) for c in object_centers]
            avg = float(np.mean(dists))
            r = avg if avg > 1e-6 else max(1.0, float(np.max(dists)))
        else:
            r = 3.0  # fallback radius

        gen = generate_camera_entries(
            axis=args.axis,
            radius=float(r),
            num_frames=args.num_frames,
            center=tuple(center.tolist()),
            up=tuple(args.up),
            start_angle_deg=args.start_angle_deg,
            image_prefix=args.image_prefix,
            start_index=cur_index,
            intrinsics=intr,
            camera_id=args.camera_id,
            model=args.model,
            rotation_conv=args.rotation_conv,
            translation_mode=args.translation_mode,
            pad=args.pad,
        )
        entries.extend(gen)

    with open(args.output, 'w') as f:
        json.dump(entries, f, indent=4)

    print(f'Wrote {len(entries)} camera entries to {args.output}')


if __name__ == '__main__':
    cli()
