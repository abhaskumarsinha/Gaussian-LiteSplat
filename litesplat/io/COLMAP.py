import os
import json
import numpy as np

def qvec2rotmat(qvec):
    """Convert quaternion [w, x, y, z] to rotation matrix."""
    w, x, y, z = qvec
    return np.array([
        [1 - 2 * (y**2 + z**2),     2 * (x*y - z*w),         2 * (x*z + y*w)],
        [2 * (x*y + z*w),           1 - 2 * (x**2 + z**2),   2 * (y*z - x*w)],
        [2 * (x*z - y*w),           2 * (y*z + x*w),         1 - 2 * (x**2 + y**2)]
    ])

def parse_cameras(file_path):
    """Parse cameras.txt from COLMAP."""
    cameras = {}
    with open(file_path, 'r') as f:
        for line in f:
            if line.startswith('#') or not line.strip():
                continue
            elems = line.strip().split()
            cam_id = int(elems[0])
            model = elems[1]
            width = float(elems[2])
            height = float(elems[3])

            if model in ['SIMPLE_PINHOLE', 'SIMPLE_RADIAL']:
                f_ = float(elems[4])
                cx, cy = float(elems[5]), float(elems[6])
                fx = fy = f_
            elif model == 'PINHOLE':
                fx, fy, cx, cy = map(float, elems[4:8])
            else:
                raise ValueError(f"Unsupported camera model: {model}")

            cameras[cam_id] = {
                "model": model,
                "width": width,
                "height": height,
                "fx": fx,
                "fy": fy,
                "cx": cx,
                "cy": cy
            }
    return cameras

def parse_images(file_path):
    """Parse images.txt from COLMAP.
    Stores camera rotations as rotation matrices (unchanged).
    """
    images = {}
    with open(file_path, 'r') as f:
        lines = f.readlines()

    i = 0
    while i < len(lines):
        line = lines[i]
        if line.startswith('#') or not line.strip():
            i += 1
            continue

        elems = line.strip().split()
        image_id = int(elems[0])
        qvec = np.array(list(map(float, elems[1:5])))
        tvec = np.array(list(map(float, elems[5:8])))
        cam_id = int(elems[8])
        name = elems[9]

        rotmat = qvec2rotmat(qvec)
        images[image_id] = {
            "name": name,
            "camera_id": cam_id,
            "rotation": rotmat.tolist(),   # keep matrices for camera.json
            "translation": tvec.tolist()
        }
        i += 2
    return images

def parse_points3D(file_path):
    """Parse points3D.txt from COLMAP and normalize colors to [0, 1].
    For gaussians.json we store 'rotation' as a quaternion [w, x, y, z].
    Since points have no orientation in COLMAP, use identity quaternion [1,0,0,0].
    """
    points = []
    if not os.path.exists(file_path):
        print("⚠️ No points3D.txt found — skipping points.")
        return points

    with open(file_path, 'r') as f:
        for line in f:
            if line.startswith('#') or not line.strip():
                continue
            elems = line.strip().split()
            xyz = list(map(float, elems[1:4]))
            rgb = np.array(list(map(float, elems[4:7]))) / 255.0

            points.append({
                "position": xyz,
                "color": rgb.tolist(),
                # store quaternion (w, x, y, z) for gaussians.json
                "rotation": [1.0, 0.0, 0.0, 0.0],
                "translation": xyz,
                "scale": [0.005, 0.005, 0.005],
                "opacity": 0.8
            })
    return points

def export_json(data, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w') as f:
        json.dump(data, f, indent=4)

def convert_colmap_to_gaussians(scene_dir, output_dir):
    """
    Convert NeRF-style COLMAP export into Gaussian + camera JSON files.

    Expected structure:
    data/<scene_name>/
    ├── images/
    │   ├── 0000.png
    │   ├── 0001.png
    │   └── ...
    ├── sparse/
    │   └── 0/
    │       ├── cameras.txt
    │       ├── images.txt
    │       ├── points3D.txt
    └── transforms.json (optional)

    Returns:
        tuple(str, str): Paths to (gaussians.json, camera.json)
    """
    sparse_dir = os.path.join(scene_dir, "sparse", "0")
    images_dir = os.path.join(scene_dir, "images")

    # --- Validate NeRF-style layout ---
    if not os.path.isdir(images_dir):
        raise FileNotFoundError(f"❌ Missing required directory: {images_dir}")
    if not os.path.isdir(sparse_dir):
        raise FileNotFoundError(f"❌ Missing required directory: {sparse_dir}")

    cameras_path = os.path.join(sparse_dir, "cameras.txt")
    images_path = os.path.join(sparse_dir, "images.txt")
    points_path = os.path.join(sparse_dir, "points3D.txt")

    for path in [cameras_path, images_path]:
        if not os.path.exists(path):
            raise FileNotFoundError(f"❌ Missing required COLMAP file: {path}")

    print(f"📁 Valid NeRF scene found at: {scene_dir}")
    print(f"  - Using COLMAP data from: {sparse_dir}")
    print(f"  - Using images from: {images_dir}")

    cameras = parse_cameras(cameras_path)
    images = parse_images(images_path)
    points = parse_points3D(points_path)

    # Merge intrinsics + extrinsics
    camera_data = []
    for img_id, img_info in images.items():
        cam_id = img_info["camera_id"]
        if cam_id not in cameras:
            print(f"⚠️ Camera {cam_id} not found for image {img_info['name']}. Skipping.")
            continue

        cam_intr = cameras[cam_id]
        camera_data.append({
            "camera_id": cam_id,
            "model": cam_intr["model"],
            "image_name": img_info["name"],
            "fx": cam_intr["fx"],
            "fy": cam_intr["fy"],
            "cx": cam_intr["cx"],
            "cy": cam_intr["cy"],
            "width": cam_intr["width"],
            "height": cam_intr["height"],
            "rotation": img_info["rotation"],    # rotation matrix (unchanged)
            "translation": img_info["translation"]
        })

    os.makedirs(output_dir, exist_ok=True)
    gaussians_path = os.path.join(output_dir, "gaussians.json")
    camera_path = os.path.join(output_dir, "camera.json")

    export_json(points, gaussians_path)
    export_json(camera_data, camera_path)

    print(f"✅ Exported:\n→ {gaussians_path}\n→ {camera_path}")
    return gaussians_path, camera_path
