from .apply_seamless_tiling_flux_model import ApplySeamlessTilingFluxModel
from .apply_seamless_tiling_vae import ApplySeamlessTilingVAE

NODE_CLASS_MAPPINGS = {
    "Apply Seamless Tiling Flux Model (Iliad)": ApplySeamlessTilingFluxModel,
    "Apply Seamless Tiling VAE (Iliad)": ApplySeamlessTilingVAE,
}

__all__ = ["NODE_CLASS_MAPPINGS"]
