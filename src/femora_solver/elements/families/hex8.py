from dataclasses import dataclass
import jax
import jax.numpy as jnp
from femora_solver.materials.linear_elastic import compute_stress
from femora_solver.elements.block import ElementBlock

# Quadrature points and weights for 2x2x2 integration
_Q = 1.0 / jnp.sqrt(3.0)
HEX8_QP = jnp.array([
    [-_Q, -_Q, -_Q],
    [ _Q, -_Q, -_Q],
    [ _Q,  _Q, -_Q],
    [-_Q,  _Q, -_Q],
    [-_Q, -_Q,  _Q],
    [ _Q, -_Q,  _Q],
    [ _Q,  _Q,  _Q],
    [-_Q,  _Q,  _Q]
])
HEX8_W = jnp.ones(8)

def hex8_shape_funcs_and_derivs(xi):
    """Returns N (8,) and dN_dxi (8, 3) for a given natural coordinate xi (3,)."""
    r, s, t = xi[0], xi[1], xi[2]
    
    N = 0.125 * jnp.array([
        (1 - r) * (1 - s) * (1 - t),
        (1 + r) * (1 - s) * (1 - t),
        (1 + r) * (1 + s) * (1 - t),
        (1 - r) * (1 + s) * (1 - t),
        (1 - r) * (1 - s) * (1 + t),
        (1 + r) * (1 - s) * (1 + t),
        (1 + r) * (1 + s) * (1 + t),
        (1 - r) * (1 + s) * (1 + t)
    ])
    
    dN_dxi = 0.125 * jnp.array([
        [-(1 - s) * (1 - t), -(1 - r) * (1 - t), -(1 - r) * (1 - s)],
        [ (1 - s) * (1 - t), -(1 + r) * (1 - t), -(1 + r) * (1 - s)],
        [ (1 + s) * (1 - t),  (1 + r) * (1 - t), -(1 + r) * (1 + s)],
        [-(1 + s) * (1 - t),  (1 - r) * (1 - t), -(1 - r) * (1 + s)],
        [-(1 - s) * (1 + t), -(1 - r) * (1 + t),  (1 - r) * (1 - s)],
        [ (1 - s) * (1 + t), -(1 + r) * (1 + t),  (1 + r) * (1 - s)],
        [ (1 + s) * (1 + t),  (1 + r) * (1 + t),  (1 + r) * (1 + s)],
        [-(1 + s) * (1 + t),  (1 - r) * (1 + t),  (1 - r) * (1 + s)]
    ])
    
    return N, dN_dxi

def hex8_element_force(u_elem, coords, params):
    """
    Computes internal force for a single Hex8 element.
    u_elem: (8, 3)
    coords: (8, 3)
    params: (2,) [E, nu]
    """
    f_int = jnp.zeros((8, 3))
    
    def qp_scan(f_int_acc, i):
        xi = HEX8_QP[i]
        w = HEX8_W[i]
        
        N, dN_dxi = hex8_shape_funcs_and_derivs(xi)
        
        # Jacobian J = sum_i dN_i/dxi * x_i
        J = dN_dxi.T @ coords # (3, 8) x (8, 3) -> (3, 3)
        detJ = jnp.linalg.det(J)
        
        # dN_dx = dN_dxi * J^{-1}
        # In JAX/NumPy, (8,3) @ (3,3) handles the mapping correctly
        J_inv = jnp.linalg.inv(J)
        dN_dx = dN_dxi @ J_inv
        
        # Compute strain B * u_elem
        # eps_xx = du/dx, eps_yy = dv/dy, eps_zz = dw/dz
        # gamma_xy = du/dy + dv/dx
        # gamma_yz = dv/dz + dw/dy
        # gamma_zx = dw/dx + du/dz
        
        eps_xx = jnp.dot(dN_dx[:, 0], u_elem[:, 0])
        eps_yy = jnp.dot(dN_dx[:, 1], u_elem[:, 1])
        eps_zz = jnp.dot(dN_dx[:, 2], u_elem[:, 2])
        gamma_xy = jnp.dot(dN_dx[:, 1], u_elem[:, 0]) + jnp.dot(dN_dx[:, 0], u_elem[:, 1])
        gamma_yz = jnp.dot(dN_dx[:, 2], u_elem[:, 1]) + jnp.dot(dN_dx[:, 1], u_elem[:, 2])
        gamma_zx = jnp.dot(dN_dx[:, 0], u_elem[:, 2]) + jnp.dot(dN_dx[:, 2], u_elem[:, 0])
        
        strain = jnp.array([eps_xx, eps_yy, eps_zz, gamma_xy, gamma_yz, gamma_zx])
        
        # Stress
        sigma = compute_stress(strain, params)
        
        # Internal force: f_int += B^T * sigma * detJ * w
        # Virtual work: f_ix = sigma_xx*dN_i/dx + tau_xy*dN_i/dy + tau_zx*dN_i/dz
        fx = dN_dx[:, 0] * sigma[0] + dN_dx[:, 1] * sigma[3] + dN_dx[:, 2] * sigma[5]
        fy = dN_dx[:, 0] * sigma[3] + dN_dx[:, 1] * sigma[1] + dN_dx[:, 2] * sigma[4]
        fz = dN_dx[:, 0] * sigma[5] + dN_dx[:, 1] * sigma[4] + dN_dx[:, 2] * sigma[2]
        
        f_qp = jnp.stack([fx, fy, fz], axis=-1) * detJ * w
        return f_int_acc + f_qp, None

    f_int, _ = jax.lax.scan(qp_scan, f_int, jnp.arange(8))
    return f_int

hex8_block_force = jax.vmap(hex8_element_force, in_axes=(0, 0, 0))

@dataclass
class Hex8Block(ElementBlock):
    coords: jax.Array # (n_elem, 8, 3) - original coordinates
    
    def compute_internal_forces(
        self,
        field_u: dict[str, jax.Array],
        history: jax.Array,
    ) -> tuple[dict[str, jax.Array], jax.Array]:
        
        u = field_u["U"]
        
        # Gather element displacements
        u_elem = u[self.connectivity] # (n_elem, 8, 3)
        
        # Handle uniform vs per-element params
        params = self.material_params
        if params.ndim == 1:
            params = jnp.broadcast_to(params, (self.connectivity.shape[0], params.shape[0]))

        # Compute element forces
        f_elem = hex8_block_force(u_elem, self.coords, params) # (n_elem, 8, 3)
        
        # Scatter to global force vector
        f_global = jnp.zeros_like(u)
        f_global = f_global.at[self.connectivity].add(f_elem)
        
        return {"U": f_global}, history

    def compute_lumped_mass(self) -> dict[str, jax.Array]:
        """
        Computes lumped mass for the block.
        Returns a dict of field_name -> (n_nodes, ncomp) mass contributions.
        """
        n_elem = self.connectivity.shape[0]
        
        params = self.material_params
        if params.ndim == 1:
            rho = params[2]
        else:
            rho = params[:, 2] # (n_elem,)
        
        # Volume of Hex8 (at each QP)
        def qp_volume(i):
            xi = HEX8_QP[i]
            w = HEX8_W[i]
            
            def elem_vol(coords):
                _, dN_dxi = hex8_shape_funcs_and_derivs(xi)
                J = dN_dxi.T @ coords
                return jnp.linalg.det(J) * w
            
            return jax.vmap(elem_vol)(self.coords)
        
        volumes = jnp.zeros(n_elem)
        for i in range(8):
            volumes += qp_volume(i)
            
        if jnp.isscalar(rho) or rho.ndim == 0:
            m_elem = (rho * volumes / 8.0).reshape(-1, 1) # (n_elem, 1)
        else:
            m_elem = (rho * volumes / 8.0).reshape(-1, 1) # (n_elem, 1)
            
        m_node_elem = jnp.broadcast_to(m_elem, (n_elem, 8)) # (n_elem, 8)
        m_node_comp = jnp.stack([m_node_elem]*3, axis=-1) # (n_elem, 8, 3)
        
        return {
            "U": (self.connectivity, m_node_comp)
        }
