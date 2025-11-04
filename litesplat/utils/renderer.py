


class Renderer(keras.layers.Layer):
    """
    Keras Renderer that aggregates outputs from multiple CameraLayer objects.

    Args:
        camera_layers (list): List of CameraLayer instances.
        default_height (int): Default image height.
        default_width (int): Default image width.
    """

    def __init__(self, camera_layers, default_height, default_width, torch_compile=False, **kwargs):
        super().__init__(**kwargs)
        self.camera_layers = camera_layers
        self.default_height = default_height
        self.default_width = default_width

        if torch_compile:
            import torch
            for layer in self.camera_layers:
                torch.compile(layer, fullgraph=True)

    def call(self, camera_ids, H=None, W=None, render=False, bsz=1024):
        """
        Args:
            camera_ids: 1D tensor/list of camera indices to render.
            H, W (optional): Image dimensions. Defaults to layer defaults.

        Returns:
            Tensor of shape (batch, H, W, 3)
        """

        # Convert inputs to tensor
        camera_ids = keras.ops.convert_to_tensor(camera_ids)
        H = H if H is not None else self.default_height
        W = W if W is not None else self.default_width

        # Pre-allocate tensor for outputs (batch, H, W, 3)
        batch_size = keras.ops.shape(camera_ids)[0]
        images = keras.ops.zeros((batch_size, H, W, 3), dtype="float32")

        # Loop over camera_ids with tensor updates
        for i in range(int(camera_ids.shape[0])):  # static small batch, okay in eager/graph
            cam_id = int(camera_ids[i])
            if cam_id >= len(self.camera_layers):
                raise ValueError(f"Camera ID {cam_id} out of range ({len(self.camera_layers)}).")

            img = self.camera_layers[cam_id](inputs=(H, W), render=render, batch_size=bsz)
            images = keras.ops.slice_update(images, (i, 0, 0, 0), img[None, ...])

        return images  # (batch, H, W, 3)
