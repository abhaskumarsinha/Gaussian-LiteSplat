import os
import json
import numpy as np

def import_gaussian_scene(scene_dir, total_gaussians: int = None, seed: int = None, preserve_order: bool = True):
    """
    Import Gaussian scene from a directory containing gaussians.json (and optionally camera.json).
    
    Args:
        scene_dir (str): Directory containing gaussians.json.
        total_gaussians (int, optional): If provided, randomly sample this number of gaussians
            (without replacement). If None, import all gaussians.
        seed (int, optional): Random seed for reproducible sampling.
        preserve_order (bool): If True (default), the sampled gaussians are returned in
            their original order in the file. If False, they are returned in random order.
    
    Returns:
        colors, positions, rotations, translations, scales, opacities (all numpy arrays, dtype=float32)
    """
    camera_path = os.path.join(scene_dir, "camera.json")
    gaussians_path = os.path.join(scene_dir, "gaussians.json")

    if not os.path.exists(gaussians_path):
        raise FileNotFoundError(f"❌ gaussians.json not found in {scene_dir}")

    print(f"📂 Importing Gaussian scene from: {scene_dir}")

    with open(gaussians_path, 'r') as f:
        gaussians = json.load(f)

    if not isinstance(gaussians, list) or len(gaussians) == 0:
        raise ValueError(f"❌ Invalid or empty gaussians.json at {gaussians_path}")

    N = len(gaussians)

    # If sampling requested, validate and pick indices
    if total_gaussians is not None:
        if not isinstance(total_gaussians, int) or total_gaussians <= 0:
            raise ValueError("total_gaussians must be a positive integer or None.")
        if total_gaussians > N:
            raise ValueError(f"Requested total_gaussians={total_gaussians} but only {N} gaussians available.")
        rng = np.random.default_rng(seed)
        sampled_idx = rng.choice(N, size=total_gaussians, replace=False)
        if preserve_order:
            sampled_idx = np.sort(sampled_idx)
        # Create a sampled list view (preserves per-gaussian grouping)
        gaussians = [gaussians[i] for i in sampled_idx]
        print(f"🎲 Randomly sampled {total_gaussians} / {N} Gaussians (seed={seed}, preserve_order={preserve_order}).")
    else:
        print(f"✅ Loaded all {N} Gaussians.")

    # Build aligned numpy arrays (i-th entry in each array corresponds to same gaussian)
    colors = np.array([g["color"] for g in gaussians], dtype=np.float32)
    positions = np.array([g["position"] for g in gaussians], dtype=np.float32)
    rotations = np.array([g.get("rotation", [0,0,0]) for g in gaussians], dtype=np.float32)
    translations = np.array([g.get("translation", [0,0,0]) for g in gaussians], dtype=np.float32)
    scales = np.array([g.get("scale", 1.0) for g in gaussians], dtype=np.float32)
    opacities = np.array([g.get("opacity", 1.0) for g in gaussians], dtype=np.float32)

    return colors, positions, rotations, translations, scales, opacities
