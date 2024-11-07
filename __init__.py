from .set_pattern_flux_model import SetPatternFluxModel
from .set_pattern_vae import SetPatternVAE

NODE_CLASS_MAPPINGS = {
    "Set Pattern Flux Model (Iliad)": SetPatternFluxModel,
    "Set Pattern VAE (Iliad)": SetPatternVAE,
}

__all__ = ["NODE_CLASS_MAPPINGS"]
