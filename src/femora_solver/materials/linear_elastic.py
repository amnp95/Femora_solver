import jax
import jax.numpy as jnp

def compute_stress(strain: jax.Array, params: jax.Array) -> jax.Array:
    """
    Computes linear elastic stress from strain.
    strain: (6,) array [eps_xx, eps_yy, eps_zz, gamma_xy, gamma_yz, gamma_zx]
    params: (2,) array [E, nu]
    """
    E = params[0]
    nu = params[1]
    
    lam = E * nu / ((1.0 + nu) * (1.0 - 2.0 * nu))
    mu = E / (2.0 * (1.0 + nu))
    
    trace_eps = strain[0] + strain[1] + strain[2]
    
    sigma = jnp.array([
        2.0 * mu * strain[0] + lam * trace_eps,
        2.0 * mu * strain[1] + lam * trace_eps,
        2.0 * mu * strain[2] + lam * trace_eps,
        mu * strain[3],
        mu * strain[4],
        mu * strain[5]
    ])
    
    return sigma
