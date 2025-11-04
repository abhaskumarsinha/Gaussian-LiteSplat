
import keras
import numpy as np
import matplotlib.pyplot as plt
from tqdm.auto import tqdm

from io import import_gaussian_scene

class GaussianParameterLayer(keras.layers.Layer):
    """
    Keras Layer that imports and holds Gaussian parameters as trainable weights.
    """

    def __init__(self, scene_dir, trainable=True, total_gaussians=None **kwargs):
        super().__init__(**kwargs)
        self.scene_dir = scene_dir
        self.trainable_flag = trainable

        colors, positions, rotations, translations, scales, opacities = import_gaussian_scene(self.scene_dir, total_gaussians=total_gaussians)

        # Add weights for each parameter
        self.colors = self.add_weight(
            name="colors",
            shape=colors.shape,
            initializer=keras.initializers.Constant(colors),
            trainable=self.trainable_flag,
        )

        self.positions = self.add_weight(
            name="positions",
            shape=positions.shape,
            initializer=keras.initializers.Constant(positions),
            trainable=self.trainable_flag,
        )

        self.rotations = self.add_weight(
            name="rotations",
            shape=rotations.shape,
            initializer=keras.initializers.Constant(rotations),
            trainable=self.trainable_flag,
        )

        self.scales = self.add_weight(
            name="scales",
            shape=scales.shape,
            initializer=keras.initializers.Constant(scales),
            trainable=self.trainable_flag,
        )

        self.opacities = self.add_weight(
            name="opacities",
            shape=opacities.shape,
            initializer=keras.initializers.Constant(opacities),
            trainable=self.trainable_flag,
        )

    def get_rotations(self):
        """Convert stored quaternions [w, x, y, z] → rotation matrices (N, 3, 3)."""

        q = keras.ops.convert_to_tensor(self.rotations)
        q = q / (keras.ops.sqrt(keras.ops.sum(q**2, axis=-1, keepdims=True)) + 1e-8)
        w, x, y, z = [q[..., i] for i in range(4)]

        r00 = 1 - 2 * (y*y + z*z)
        r01 = 2 * (x*y - z*w)
        r02 = 2 * (x*z + y*w)

        r10 = 2 * (x*y + z*w)
        r11 = 1 - 2 * (x*x + z*z)
        r12 = 2 * (y*z - x*w)

        r20 = 2 * (x*z - y*w)
        r21 = 2 * (y*z + x*w)
        r22 = 1 - 2 * (x*x + y*y)

        return keras.ops.stack([
            keras.ops.stack([r00, r01, r02], axis=-1),
            keras.ops.stack([r10, r11, r12], axis=-1),
            keras.ops.stack([r20, r21, r22], axis=-1),
        ], axis=-2)

    def call(inputs):
        return inputs
