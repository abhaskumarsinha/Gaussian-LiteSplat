from keras import ops
from PIL import Image

import keras
from keras import ops
import numpy as np
import matplotlib.pyplot as plt
from tqdm.auto import tqdm
import os
import json



class CameraLayer(keras.layers.Layer):
    """
    Camera layer that selects a single camera by index and computes
    Jacobians, projection matrix, and scaling matrices in TensorFlow.
    """

    def __init__(self,
                 scene_dir,
                 gaussians,
                 images_dir = None,
                 trainable_focus=False,
                 trainable_principal=False,
                 trainable_extrinsics=False,
                 camera_index=0,
                 max_gaussians=None,
                 output_h=None,
                 output_w=None,
                 **kwargs):

        super().__init__(**kwargs)

        self.scene_dir = scene_dir
        self.trainable_focus = trainable_focus
        self.trainable_principal = trainable_principal
        self.trainable_extrinsics = trainable_extrinsics
        self.camera_index = camera_index
        self.gaussians = gaussians
        self.max_gaussians = max_gaussians

        camera_path = os.path.join(self.scene_dir, "camera.json")
        with open(camera_path, "r") as f:
            cameras = json.load(f)

        if self.camera_index >= len(cameras):
            raise ValueError(f"Camera index {self.camera_index} out of range ({len(cameras)} cameras found).")

        cam = cameras[self.camera_index]

        # Store metadata
        self.camera_id = cam["camera_id"]
        self.model = cam["model"]
        self.image_name = cam["image_name"]
        self.width = cam["width"]
        self.height = cam["height"]

        if images_dir is not None:
            path = os.path.join(images_dir, self.image_name)
            image = Image.open(path).convert("RGB")
            if output_h and output_w is None:
                image = image.resize((self.width, self.height))
            else:
                image = image.resize((output_w, output_h))
            image = image.transpose(Image.FLIP_TOP_BOTTOM)
            self.y_real = keras.ops.convert_to_tensor(image)

        # Store numeric parameters as constants
        self.fx = self.add_weight(
            name="focal_length_x",
            shape=keras.ops.convert_to_tensor(cam["fx"]).shape,
            initializer=keras.initializers.Constant(cam["fx"]),
            trainable=self.trainable_focus,
        )
        self.fy = self.add_weight(
            name="focal_length_y",
            shape=keras.ops.convert_to_tensor(cam["fy"]).shape,
            initializer=keras.initializers.Constant(cam["fy"]),
            trainable=self.trainable_focus,
        )
        self.cx = self.add_weight(
            name="principal_point",
            shape=keras.ops.convert_to_tensor(cam["cx"]).shape,
            initializer=keras.initializers.Constant(cam["cx"]),
            trainable=self.trainable_principal,
        )
        self.cy = self.add_weight(
            name="principal_point",
            shape=keras.ops.convert_to_tensor(cam["cy"]).shape,
            initializer=keras.initializers.Constant(cam["cy"]),
            trainable=self.trainable_principal,
        )
        self.rotations = self.add_weight(
            name="rotation_matrix",
            shape=keras.ops.convert_to_tensor(cam["rotation"]).shape,
            initializer=keras.initializers.Constant(keras.ops.convert_to_tensor(cam["rotation"])),
            trainable=self.trainable_extrinsics,
        )

        self.translations = self.add_weight(
            name="translation_vector",
            shape=keras.ops.convert_to_tensor(cam["translation"]).shape,
            initializer=keras.initializers.Constant(keras.ops.convert_to_tensor(cam["translation"])),
            trainable=self.trainable_extrinsics,
        )

        self.scaling_matrices = self.make_scaling_matrices(self.gaussians.scales)
        #self.projection_matrix = self.get_projection_matrix(keras.ops.convert_to_tensor(self.fx),
        #                                                    keras.ops.convert_to_tensor(self.fy),
        #                                                    keras.ops.convert_to_tensor(self.cx),
        #                                                    keras.ops.convert_to_tensor(self.cy))

        self.get_sorted_keys()


    def get_sorted_keys(self):
        reference_vector = keras.ops.convert_to_tensor(self.translations)

        diffs = self.gaussians.positions - reference_vector
        distances = keras.ops.sqrt(keras.ops.sum(diffs**2, axis=-1))
        sorted_indices = keras.ops.argsort(distances, axis=0)
        self.sorted_indices = sorted_indices



    # ---------------------------------------------------------
    # --- Keras version of make_scaling_matrices() ------------
    # ---------------------------------------------------------
    def make_scaling_matrices(self, scales):
        """
        TensorFlow/Keras version of make_scaling_matrices(scales).

        Args:
            scales: (N, 3) tensor of per-axis scales
        Returns:
            (N, 3, 3) tensor of diagonal scaling matrices
        """
        scales = keras.ops.convert_to_tensor(scales, dtype='float32')
        n = keras.ops.shape(scales)[0]

        # Create a single 3x3 identity matrix
        eye = keras.ops.eye(3, dtype='float32')  # shape (3, 3)

        # Tile across the batch dimension -> (N, 3, 3)
        eye = keras.ops.tile(keras.ops.expand_dims(eye, 0), (n, 1, 1))

        # Multiply each batch with its scale vector -> broadcast (N, 3, 3)
        return eye * keras.ops.expand_dims(scales, -1)


    # ---------------------------------------------------------
    # --- Keras version of projection_jacobians() -------------
    # ---------------------------------------------------------
    def projection_jacobians(self, fx, fy, positions):
        """
        TensorFlow version of projection_jacobians().
        Args:
            fx, fy: scalar tensors
            positions: (N, 3)
        Returns:
            (N, 2, 3)
        """
        X = positions[:, 0]
        Y = positions[:, 1]
        Z = keras.ops.maximum(positions[:, 2], 1e-8)

        zeros = keras.ops.zeros_like(Z)

        Jx = keras.ops.stack([
            fx / Z,
            zeros,
            -fx * X / (Z ** 2)
        ], axis=1)

        Jy = keras.ops.stack([
            zeros,
            fy / Z,
            -fy * Y / (Z ** 2)
        ], axis=1)

        return keras.ops.stack([Jx, Jy], axis=1)  # (N, 2, 3)

    # ---------------------------------------------------------
    # --- Keras version of get_projection_matrix() ------------
    # ---------------------------------------------------------
    def get_projection_matrix(self, fx, fy, cx, cy, R=None, t=None):
        """
        TensorFlow version of get_projection_matrix().
        Args:
            fx, fy, cx, cy: scalars
            R: (3, 3)
            t: (3,)
        Returns:
            (3, 3) or (3, 4)
        """
        fx = keras.ops.convert_to_tensor(fx)
        fy = keras.ops.convert_to_tensor(fy)
        cx = keras.ops.convert_to_tensor(cx)
        cy = keras.ops.convert_to_tensor(cy)
        K = keras.ops.convert_to_tensor([
            [fx, 0.0, cx],
            [0.0, fy, cy],
            [0.0, 0.0, 1.0]
        ], dtype='float32')

        if R is None or t is None:
            return K

        R = keras.ops.reshape(keras.ops.convert_to_tensor(R, dtype='float32'), (3, 3))
        t = keras.ops.reshape(keras.ops.convert_to_tensor(t, dtype='float32'), (3, 1))
        Rt = keras.ops.concat([R, t], axis=1)

        return keras.ops.matmul(K, Rt)  # (3, 4)


    def alpha_blend(self, colors, opacities, weights, *,
                          closest_first=True,
                          clip=True,
                          dtype='float32'):
        """
        Keras/Core version of alpha_blend.

        Parameters
        ----------
        colors : (N, 3) tensor/array
        opacities : (N,) or (N,1) tensor/array
        weights : (P, N) or (N,) tensor/array
        closest_first : bool
            If True, index 0 is treated as closest (applied first).
            If False, order is reversed.
        clip : bool
            Clip effective alpha to [0,1]
        dtype : str or tf.DType
            Numeric dtype to use internally.

        Returns
        -------
        final_colors : (P, 3) tensor
        final_opacities : (P,) tensor
        """
        # Convert to tensors with desired dtype
        colors = ops.convert_to_tensor(colors, dtype=dtype)           # (N, 3)
        opacities = ops.convert_to_tensor(opacities, dtype=dtype)    # (N,) or (N,1)
        weights = ops.convert_to_tensor(weights, dtype=dtype)        # (P, N) or (N,)


        # Get shapes
        P = ops.shape(weights)[0]
        N = ops.shape(weights)[1]

        # Static sanity check where possible (best-effort)
        # If static shapes are known, validate sizes
        if colors.shape[0] is not None and colors.shape[0] != N:
            raise ValueError(f"Mismatch: weights has N={N}, but colors has shape {colors.shape}")
        if opacities.shape[0] is not None and opacities.shape[0] != N:
            raise ValueError(f"Mismatch: weights has N={N}, but opacities has shape {opacities.shape}")

        # Optionally reverse order (use gather with reversed indices)
        if not closest_first:
            idx = ops.arange(N - 1, -1, -1)     # indices: N-1, ..., 0
            colors = ops.gather(colors, idx)
            opacities = ops.gather(opacities, idx)
            weights = ops.gather(weights, idx, axis=1)

        # effective alpha a[p,i] = weights[p,i] * opacities[i] -> shape (P, N)
        a = weights * ops.expand_dims(opacities, 0)   # broadcast opacities across P

        if clip:
            a = ops.clip(a, 0.0, 1.0)

        # one_minus = 1 - a  (P, N)
        one_minus = 1.0 - a

        # cumprod inclusive along axis=1 -> prod_{j<=i} (1 - a_j)
        # ops.cumprod supports `axis` argument
        cumprod_inclusive = ops.cumprod(one_minus, axis=1)

        # T_prev is exclusive product: shift right and prepend ones: shape (P, N)
        ones_col = ops.ones((P, 1), dtype=dtype)
        # cumprod_inclusive[:, :-1] (may be empty if N==1)
        if ops.shape(cumprod_inclusive)[1] > 1:
            prev = ops.slice(cumprod_inclusive, (0, 0), (P, ops.shape(cumprod_inclusive)[1] - 1))
            T_prev = ops.concatenate([ones_col, prev], axis=1)
        else:
            # N == 1 case
            T_prev = ones_col  # shape (P,1)

        # weights_for_color = (a * T_prev)[..., None]  -> (P, N, 1)
        weights_for_color = ops.expand_dims(a * T_prev, -1)

        # contributions = weights_for_color * colors[None, :, :] -> (P, N, 3)
        contributions = weights_for_color * ops.expand_dims(colors, 0)

        # final_colors = contributions.sum(axis=1) -> (P, 3)
        final_colors = ops.sum(contributions, axis=1)

        # transmittance_final = cumprod_inclusive[:, -1] -> (P,)
        transmittance_final = ops.slice(cumprod_inclusive, (0, ops.shape(cumprod_inclusive)[1] - 1), (P, 1))
        transmittance_final = ops.reshape(transmittance_final, (P,))

        final_opacities = 1.0 - transmittance_final  # (P,)

        return final_colors

    def render_pixel_color(self, XY):
        """
        Keras version of render_pixel_color().
        Args:
            XY: (N, 2) tensor of pixel coordinates
            cam: dict-like with fx, fy, cx, cy, height, rotation, translation
            positions, rotations, scales, colors, opacities: tensors
        Returns:
            colors_out: (N, 3)
        """

        fx = keras.ops.convert_to_tensor(self.fx)
        fy = keras.ops.convert_to_tensor(self.fy)
        cx = keras.ops.convert_to_tensor(self.cx)
        cy = keras.ops.convert_to_tensor(self.cy)
        height = self.height
        R = keras.ops.convert_to_tensor(self.rotations)
        t = keras.ops.convert_to_tensor(self.translations)


        # Step 1: Projection matrix
        proj = self.get_projection_matrix(fx, fy, cx, cy)

        # Step 2: Scaling matrices
        scales_matrices = keras.ops.exp(self.scaling_matrices)

        # Step 2: Project Gaussians to 2D
        mu_cam = keras.ops.einsum('ij,bj->bi', R, self.gaussians.positions) + t
        P_mu = keras.ops.einsum('ij,bj->bi', proj, mu_cam)
        jacobians = self.projection_jacobians(fx, fy, mu_cam)
        P_mu_2d = P_mu[:, :2] / P_mu[:, 2:3]
        P_mu_2d[:, 1] = height - P_mu_2d[:, 1]

        # ------------------ SPLIT ---------------------------

        # Step 3: Compute XY deltas
        delta = XY[:, None, :] - P_mu_2d[None, :, :] # (Batch of xy points, gaussians, 2)

        # Step 4: Combine rotation, scale, and Jacobian
        net_rot = keras.ops.einsum('ij, bji -> bij', keras.ops.array(R), self.gaussians.get_rotations())
        T = keras.ops.einsum('bri, bis -> brs', net_rot, scales_matrices)
        T = keras.ops.einsum('bij, bjk -> bik', jacobians, T)   # T -> ( Number of Gaussians, 2, 3)

        # Step 5: Covariance and its inverse
        Sigma2D = keras.ops.einsum('bij,bkj->bik', T, T) # Sigma2D -> (Number of Gaussians, 2, 2)
        Sigma_inv = keras.ops.inv(Sigma2D)
        # print trace of Sigma2D for the two gaussians you showed

        # Step 6: Mahalanobis distance and weights
        mahalanobis = keras.ops.einsum('nmi,mij,nmj->nm', delta, Sigma_inv, delta) # mahalanobis -> (Batch of xy points, 1)
        weights = keras.ops.exp(-0.5 * mahalanobis)

        # Step 7: Prepare opacities and composite
        sorted_gaussians = keras.ops.take(self.gaussians.colors, self.sorted_indices, axis=0)[:self.max_gaussians]
        sorted_opacities = keras.ops.take(self.gaussians.opacities, self.sorted_indices, axis=0)[:self.max_gaussians]
        sorted_weights = keras.ops.take(weights, self.sorted_indices, axis=-1)[:, :self.max_gaussians]

        # Step 8: Alpha Rendering
        colors_out = self.alpha_blend(sorted_gaussians, sorted_opacities, sorted_weights)

        return colors_out

    # ---------------------------------------------------------
    # --- Call: combine everything ----------------------------
    # ---------------------------------------------------------
    def call(self, inputs, batch_size=1024, render=True):
        # Step 1: compute sampling step
        target_H, target_W = inputs
        full_W, full_H = self.width, self.height
        step_y = full_H / target_H
        step_x = full_W / target_W

        # Step 2: create a downsampled pixel grid
        ys = keras.ops.arange(0, full_H, step_y)
        xs = keras.ops.arange(0, full_W, step_x)
        ys, xs = keras.ops.meshgrid(ys, xs, indexing="ij")
        XY_all = keras.ops.stack([xs, ys], axis=-1).reshape(-1, 2)

        # Step 3: prepare empty buffer
        rendered_colors = keras.ops.zeros((len(XY_all), 3), dtype='float32')

        # Step 4: render each batch
        for i in range(0, len(XY_all), batch_size):
            XY_batch = XY_all[i:i + batch_size]
            colors_out = self.render_pixel_color(XY_batch)
            rendered_colors[i:i + batch_size] = colors_out

        target_H, target_W = int(target_H), int(target_W)
        # Step 5: reshape to (target_H, target_W, 3)
        image = keras.ops.reshape(rendered_colors, (target_H, target_W, 3))

        if render:
            plt.figure(figsize=(6, 5))
            plt.imshow(np.clip(image, 0, 1))
            plt.axis("off")
            plt.title(f"Downsampled Render ({target_W}×{target_H})")
            plt.show()
        else:
            return image
