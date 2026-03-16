import jax
import jax.numpy as jnp
from dataclasses import dataclass
from femora_solver.elements.block import ElementBlock

def beam2_element_force(u_elem, r_elem, coords, params, section_params):
    """
    Simple 3D linear beam element internal force.
    u_elem: (2, 3) - nodal displacements
    r_elem: (2, 3) - nodal rotations
    coords: (2, 3) - nodal coordinates
    params: (3,) [E, nu, rho]
    section_params: (4,) [Area, Iy, Iz, J]
    """
    E = params[0]
    nu = params[1]
    G = E / (2.0 * (1.0 + nu))
    
    A, Iy, Iz, J = section_params
    
    p1, p2 = coords[0], coords[1]
    L = jnp.linalg.norm(p2 - p1)
    
    # Local axis x
    ex = (p2 - p1) / L
    
    # Local axis y (assuming vertical for now)
    v_up_default = jnp.array([0.0, 0.0, 1.0])
    v_up_alt = jnp.array([0.0, 1.0, 0.0])
    
    is_vertical = jnp.abs(jnp.dot(ex, v_up_default)) > 0.99
    v_up = jnp.where(is_vertical, v_up_alt, v_up_default)
    
    ez = jnp.cross(ex, v_up)
    ez = ez / jnp.linalg.norm(ez)
    ey = jnp.cross(ez, ex)
    
    # Transformation matrix R (3, 3)
    R_mat = jnp.stack([ex, ey, ez]) # local to global
    
    # Nodal displacement in local coords
    u_loc = (u_elem @ R_mat.T) # (2, 3) -> [u, v, w]
    r_loc = (r_elem @ R_mat.T) # (2, 3) -> [th_x, th_y, th_z]
    
    # Extract components
    u1, v1, w1 = u_loc[0]
    u2, v2, w2 = u_loc[1]
    thx1, thy1, thz1 = r_loc[0]
    thx2, thy2, thz2 = r_loc[1]
    
    # 1. Axial and Torsion
    f_axial = (E * A / L) * (u2 - u1)
    m_torque = (G * J / L) * (thx2 - thx1)
    
    # 2. Bending in X-Y plane (v, th_z)
    fy1 = (E * Iz / L**3) * (12.0 * (v1 - v2) + 6.0 * L * (thz1 + thz2))
    fy2 = -fy1
    mz1 = (E * Iz / L**3) * (6.0 * L * (v1 - v2) + 4.0 * L**2 * thz1 + 2.0 * L**2 * thz2)
    mz2 = (E * Iz / L**3) * (6.0 * L * (v1 - v2) + 2.0 * L**2 * thz1 + 4.0 * L**2 * thz2)
    
    # 3. Bending in X-Z plane (w, th_y)
    fz1 = (E * Iy / L**3) * (12.0 * (w1 - w2) - 6.0 * L * (thy1 + thy2))
    fz2 = -fz1
    my1 = (E * Iy / L**3) * (-6.0 * L * (w1 - w2) + 4.0 * L**2 * thy1 + 2.0 * L**2 * thy2)
    my2 = (E * Iy / L**3) * (-6.0 * L * (w1 - w2) + 2.0 * L**2 * thy1 + 4.0 * L**2 * thy2)

    # Return forces in local coords
    f_u_loc = jnp.array([
        [-f_axial, fy1, fz1],
        [ f_axial, fy2, fz2]
    ])
    f_r_loc = jnp.array([
        [m_torque, my1, mz1],
        [-m_torque, my2, mz2]
    ])
    
    # Transform back to global
    f_u_glob = f_u_loc @ R_mat
    f_r_glob = f_r_loc @ R_mat
    
    return f_u_glob, f_r_glob

beam2_block_force = jax.vmap(beam2_element_force, in_axes=(0, 0, 0, 0, 0))

@dataclass
class Beam2Block(ElementBlock):
    coords: jax.Array # (n_elem, 2, 3)
    section_params: jax.Array # (n_elem, 4) [A, Iy, Iz, J]
    
    def compute_internal_forces(
        self,
        field_u: dict[str, jax.Array],
        history: jax.Array,
    ) -> tuple[dict[str, jax.Array], jax.Array]:
        u = field_u["U"]
        r = field_u["R"]
        
        u_elem = u[self.connectivity] # (n_elem, 2, 3)
        r_elem = r[self.connectivity] # (n_elem, 2, 3)
        
        params = self.material_params
        if params.ndim == 1:
            params = jnp.broadcast_to(params, (self.connectivity.shape[0], params.shape[0]))
            
        sec_params = self.section_params
        if sec_params.ndim == 1:
            sec_params = jnp.broadcast_to(sec_params, (self.connectivity.shape[0], sec_params.shape[0]))

        f_u_elem, f_r_elem = beam2_block_force(u_elem, r_elem, self.coords, params, sec_params)
        
        f_u_glob = jnp.zeros_like(u)
        f_r_glob = jnp.zeros_like(r)
        
        f_u_glob = f_u_glob.at[self.connectivity].add(f_u_elem)
        f_r_glob = f_r_glob.at[self.connectivity].add(f_r_elem)
        
        return {"U": f_u_glob, "R": f_r_glob}, history

    def compute_lumped_mass(self) -> dict[str, jax.Array]:
        n_elem = self.connectivity.shape[0]
        rho = self.material_params[:, 2] if self.material_params.ndim > 1 else self.material_params[2]
        
        sec_params = self.section_params
        if sec_params.ndim == 1:
            A, Iy, Iz, J = sec_params[0], sec_params[1], sec_params[2], sec_params[3]
        else:
            A, Iy, Iz, J = sec_params[:, 0], sec_params[:, 1], sec_params[:, 2], sec_params[:, 3]
        
        p1, p2 = self.coords[:, 0], self.coords[:, 1]
        L = jnp.linalg.norm(p2 - p1, axis=-1)
        
        m_total = rho * A * L
        m_node = (m_total / 2.0).reshape(-1, 1)
        m_node_elem = jnp.broadcast_to(m_node, (n_elem, 2))
        m_u_comp = jnp.stack([m_node_elem]*3, axis=-1)
        
        # Rotational mass (rho * I * L / 2)
        m_ry = (rho * Iy * L / 2.0).reshape(-1, 1)
        m_rz = (rho * Iz * L / 2.0).reshape(-1, 1)
        m_rx = (rho * J * L / 2.0).reshape(-1, 1)
        
        m_r = jnp.stack([m_rx, m_ry, m_rz], axis=-1) # (n_elem, 1, 3)
        m_r_elem = jnp.broadcast_to(m_r, (n_elem, 2, 3))
        
        return {
            "U": (self.connectivity, m_u_comp),
            "R": (self.connectivity, m_r_elem)
        }
