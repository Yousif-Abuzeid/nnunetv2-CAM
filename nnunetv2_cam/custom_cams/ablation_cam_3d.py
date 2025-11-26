import torch
from pytorch_grad_cam.ablation_cam import AblationCAM
from pytorch_grad_cam.ablation_layer import AblationLayer


class AblationLayer3D(AblationLayer):
    def __init__(self):
        super(AblationLayer3D, self).__init__()

    def set_next_batch(self, input_batch_index, activations, num_channels_to_ablate):
        """
        This creates the next batch of activations from the layer.
        Just take corresponding batch member from activations, and repeat it num_channels_to_ablate times.
        """
        if len(activations.shape) == 5:
            # 3D case: (Batch, Channels, Depth, Height, Width)
            self.activations = (
                activations[input_batch_index, :, :, :, :]
                .clone()
                .unsqueeze(0)
                .repeat(num_channels_to_ablate, 1, 1, 1, 1)
            )
        else:
            # 2D case: (Batch, Channels, Height, Width)
            self.activations = (
                activations[input_batch_index, :, :, :]
                .clone()
                .unsqueeze(0)
                .repeat(num_channels_to_ablate, 1, 1, 1)
            )

    def __call__(self, x):
        output = self.activations
        if output is None:
            return x

        # If we are ablating, we need to make sure the output has the same shape as x
        # This is important when x is a batch of images, and we are ablating one of them.
        # In that case, self.activations will have the shape of the batch of ablated images.
        if output.shape != x.shape:
            # This happens when we are not ablating, but just passing through
            # or when we are ablating a different batch size than the input
            return x

        for i in range(output.size(0)):
            # In the 3D case, we need to handle the extra dimension
            if len(output.shape) == 5:
                # (Batch, Channels, Depth, Height, Width)
                for j in range(output.size(1)):
                    if self.indices[i] == j:
                        output[i, j, :, :, :] = 0
            else:
                # (Batch, Channels, Height, Width)
                for j in range(output.size(1)):
                    if self.indices[i] == j:
                        output[i, j, :, :] = 0
        return output


class AblationCAM3D(AblationCAM):
    def __init__(
        self,
        model,
        target_layers,
        use_cuda=False,
        reshape_transform=None,
        compute_input_gradient=False,
        uses_gradients=False,
        ablation_layer=AblationLayer3D(),
        ratio_channels_to_ablate=1.0,
    ):
        super(AblationCAM3D, self).__init__(
            model,
            target_layers,
            use_cuda,
            reshape_transform,
            compute_input_gradient,
            uses_gradients,
            ablation_layer,
            ratio_channels_to_ablate,
        )
