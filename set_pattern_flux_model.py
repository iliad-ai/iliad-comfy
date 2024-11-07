import torch


class SetPatternFluxModel:
    def __init__(self):
        self.i = 0

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

    def apply(self, model, bypass):
        if bypass:
            return (model,)

        original_forward = model.model.diffusion_model.forward

        def pattern_forward(x, timestep, context, y, guidance, control=None, **kwargs):
            shift_h = 0
            shift_w = 0
            bs, c, h, w = x.shape
            if timestep[0] > 0.50:  # 0.25
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
                result = original_forward(
                    rolled_x, timestep, context, y, guidance, control, **kwargs
                )

                # Unshift and return
                return torch.roll(result, shifts=(-shift_h, -shift_w), dims=(2, 3))
            else:
                # First forward pass result, without any shift
                r0 = original_forward(
                    x, timestep, context, y, guidance, control, **kwargs
                )

                # Second forward pass result, with half-shift in both dimensions
                half_shifted_x = torch.roll(x, shifts=(h // 2, w // 2), dims=(2, 3))
                r1 = original_forward(
                    half_shifted_x, timestep, context, y, guidance, control, **kwargs
                )
                r1 = torch.roll(
                    r1, shifts=(-h // 2, -w // 2), dims=(2, 3)
                )  # Unshift r1 back

                # Initialize a mask that matches the shape of the latents
                mask = torch.ones(bs, c, h, w, device=x.device)
                # Make the mask 0 in the seam areas
                # (16 pixels deep on each side due to 8x compression image->latent)
                mask[:, :, :2, :] = 0  # Top edge
                mask[:, :, -2:, :] = 0  # Bottom edge
                mask[:, :, :, :2] = 0  # Left edge
                mask[:, :, :, -2:] = 0  # Right edge

                # Blend r0 and r1 using the mask
                result = mask * r0 + (1 - mask) * r1

                return result

        # Monkey-patch the forward method
        model.model.diffusion_model.forward = lambda *args, **kwargs: pattern_forward(
            *args, **kwargs
        )

        return (model,)


NODE_CLASS_MAPPINGS = {"SetPatternFluxModel": SetPatternFluxModel}

NODE_DISPLAY_NAME_MAPPINGS = {"SetPatternFluxModel": "Set Pattern Flux Model (Iliad)"}
