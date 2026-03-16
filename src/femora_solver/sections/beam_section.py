from dataclasses import dataclass
import jax.numpy as jnp

@dataclass
class BeamSection:
    area: float
    iy: float
    iz: float
    j: float
    # Orientation vector (e.g., local y-axis in global coords)
    vec_y: jnp.ndarray
