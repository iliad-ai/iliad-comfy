import torch


class SetPatternVAE:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "vae": ("VAE",),
                "bypass": ("BOOLEAN", {"default": False}),
            },
        }

    RETURN_TYPES = ("VAE",)
    FUNCTION = "apply"
    CATEGORY = "latent"

    def apply(self, vae, bypass):
        if bypass:
            return (vae,)

        original_decode = vae.decode

        def new_decode(samples_in):
            bs, c, h, w = samples_in.shape

            d0 = original_decode(samples_in)
            d1 = original_decode(
                torch.roll(samples_in, shifts=(h // 2, w // 2), dims=(2, 3))
            )

            dbs, dh, dw, dc = d0.shape

            d1 = torch.roll(d1, shifts=(-dh // 2, -dw // 2), dims=(1, 2))

            # Initialize mask with ones (center is fully from d0)
            mask = torch.ones(dbs, dh, dw, 1, device=samples_in.device)

            # Apply the hard seam areas (16 pixels deep)
            mask[:, :16, :, :] = 0  # top
            mask[:, -16:, :, :] = 0  # bottom
            mask[:, :, :16, :] = 0  # left
            mask[:, :, -16:, :] = 0  # right

            # Blend d0 and d1 using the mask
            result = mask * d0 + (1 - mask) * d1

            return result

        # Monkey-patch the decode method
        vae.decode = lambda *args, **kwargs: new_decode(*args, **kwargs)

        return (vae,)


NODE_CLASS_MAPPINGS = {"SetPatternVAE": SetPatternVAE}

NODE_DISPLAY_NAME_MAPPINGS = {"SetPatternVAE": "Set Pattern VAE (Iliad)"}
