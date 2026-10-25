"""Create the selected S3TO model without changing the existing MLP default."""

from fl.model import SimpleMLP
from cnn_foundation.src.s3to.model import MNISTCNN


def create_model(model_name: str = "mlp"):
    """Return a fresh model instance for 'mlp' or 'cnn'."""
    name = model_name.strip().lower()

    if name == "mlp":
        return SimpleMLP()
    if name == "cnn":
        return MNISTCNN()

    raise ValueError(
        f"Unsupported model {model_name!r}. Choose 'mlp' or 'cnn'."
    )
