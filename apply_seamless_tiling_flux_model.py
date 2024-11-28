import torch


class ApplySeamlessTilingFluxModel:
    def __init__(self):
        self.i = 0
        self.original_forward = None

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "bypass": ("BOOLEAN", {"default": False}),
            },
        }

    RETURN_TYPES = ("MODEL",)
    FUNCTION = "apply"
    CATEGORY = "latent"
    OUTPUT_NODE = True

    def apply(self, model, bypass):
        if self.original_forward is None:
            self.original_forward = model.model.diffusion_model.forward

        if bypass:
            model.model.diffusion_model.forward = self.original_forward
            return (model,)

        def seamless_forward(x, timestep, context, y, guidance, control=None, **kwargs):
            shift_h = 0
            shift_w = 0
            bs, c, h, w = x.shape

            # 0.5 because, beyond this (0.5 -> 0.0), there are visible seams
            # around where the shifts occurred.
            # So, to get around it, we do two passes past 0.5
            # and merge them (see else clause).
            if timestep[0] > 0.50:
                shift_schedule = [
                    (h // 16, w // 16),
                    (-h // 16, -w // 16),
                    (h // 16, -w // 16),
                    (-h // 16, w // 16),
                ]
                shift_h = shift_schedule[self.i % 4][0]
                shift_w = shift_schedule[self.i % 4][1]
                # randomly set these between 1x to 2x to decrease the likelihood
                # of getting a seam in the same place
                shift_h = int(shift_h * (1.0 + torch.rand(1).item()))
                shift_w = int(shift_w * (1.0 + torch.rand(1).item()))
                self.i += 1

                # Shift
                rolled_x = torch.roll(x, shifts=(shift_h, shift_w), dims=(2, 3))

                # Call the original forward method
                result = self.original_forward(
                    rolled_x, timestep, context, y, guidance, control, **kwargs
                )

                # Unshift and return
                x = torch.roll(result, shifts=(-shift_h, -shift_w), dims=(2, 3))
            else:
                # First forward pass result, without any shift
                r0 = self.original_forward(
                    x, timestep, context, y, guidance, control, **kwargs
                )

                # Second forward pass result, with half-shift in both dimensions
                half_shifted_x = torch.roll(x, shifts=(h // 2, w // 2), dims=(2, 3))
                r1 = self.original_forward(
                    half_shifted_x, timestep, context, y, guidance, control, **kwargs
                )
                # Unshift r1
                r1 = torch.roll(r1, shifts=(-h // 2, -w // 2), dims=(2, 3))

                # Initialize a mask that matches the shape of the latents
                mask = torch.ones(bs, c, h, w, device=x.device)
                # Make the mask 0 in the seam areas
                # (16 pixels deep on each side due to 8x compression image -> latent)
                mask[:, :, :2, :] = 0  # Top edge
                mask[:, :, -2:, :] = 0  # Bottom edge
                mask[:, :, :, :2] = 0  # Left edge
                mask[:, :, :, -2:] = 0  # Right edge

                # Blend r0 and r1 using the mask
                result = mask * r0 + (1 - mask) * r1

                x = result

            return x

        # Monkey-patch the forward method
        model.model.diffusion_model.forward = lambda *args, **kwargs: seamless_forward(
            *args, **kwargs
        )

        return (model,)

    @classmethod
    def IS_CHANGED(*args, **kwargs):
        return float("NaN")


NODE_CLASS_MAPPINGS = {"ApplySeamlessTilingFluxModel": ApplySeamlessTilingFluxModel}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ApplySeamlessTilingFluxModel": "Apply Seamless Tiling Flux Model (Iliad)"
}
