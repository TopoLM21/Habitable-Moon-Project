"""Keep saved Genesis histories tied to the thermal law that produced them."""
from .genesis import MODEL_VERSION


def require_thermal_model_version(metadata):
    """Reject legacy histories before dataclass defaults can change their law."""
    saved = metadata.get("thermal_model_version")
    if saved != MODEL_VERSION:
        raise ValueError(
            f"Genesis thermal model mismatch: checkpoint uses {saved or 'legacy/unversioned'}, "
            f"current model is {MODEL_VERSION}. Regenerate the Genesis/Starter run from "
            "its initial conditions; old thermal histories cannot resume under a different heat-transfer law."
        )
