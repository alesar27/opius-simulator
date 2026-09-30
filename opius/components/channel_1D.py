import numpy as np
from opius.physics.physics_helper import Colebrook, Reynolds

class Channel1D:
    def __init__(self, name, length, nx, area, theta, roughness, coolant, dh=None):
        self.name = name
        self.L = float(length)
        self.nx = int(nx)
        self.dx = self.L / (self.nx - 1)
        self.x = np.linspace(0, self.L, self.nx)
        self.coolant = coolant
        self.roughness = roughness
        self.A = np.full(self.nx, float(area)) if np.isscalar(area) else np.array(area, dtype=float)
        self.theta = np.full(self.nx, float(theta)) if np.isscalar(theta) else np.array(theta, dtype=float)
        self.Dh = np.sqrt(4.0 * self.A / np.pi) if dh is None else (
            np.full(self.nx, float(dh)) if np.isscalar(dh) else np.array(dh, dtype=float)
        )
        # inertia
        self.inertia = float(np.sum(self.dx / self.A))
        self.i = np.zeros(self.nx)
        self.Cb = np.zeros(self.nx)
        self.T = np.zeros(self.nx)
        self.rho = np.zeros(self.nx)

    def set_initial_profile(self, i_initial, Cb_initial):
        self.i = i_initial.copy()
        self.Cb = Cb_initial.copy()
        self.T[:] = self.coolant.T_h(self.i)
        self.rho[:] = self.coolant.rho(self.T)

    def upwind_scheme (self, m_dot, dt, source_power = None, source_boron = None): # Creates the array for the next time step using an upwind scheme
        if source_power is None: source_power = np.zeros(self.nx)
        if source_boron is None: source_boron = np.zeros(self.nx)
        i_new = self.i.copy()
        Cb_new = self.Cb.copy()
        v = m_dot / (self.rho * self.A)
        direction = np.sign(v[0])
        CFL = abs(v * dt / self.dx)
        if np.any(CFL > 1):
            print(f"Upwind scheme is unstable! Max CFL = {CFL.max():.4f} in {self.name}")
        else :
            if direction > 0 :
                for k in range(1, self.nx):
                    i_new[k] = (source_power[k] / (self.rho[k]) - v[k] / self.dx * (self.i[k] - self.i[k-1])) * dt + self.i[k]
                    Cb_new[k] = (source_boron[k] - (v[k] / self.dx * (self.Cb[k] - self.Cb[k-1]))) * dt + self.Cb[k]
            else: 
                for k in range(self.nx - 2, -1, -1):
                    i_new[k] = (source_power[k] / (self.rho[k]) - v[k] / self.dx * (self.i[k+1] - self.i[k])) * dt + self.i[k]
                    Cb_new[k] = (source_boron[k] - (v[k] / self.dx * (self.Cb[k+1] - self.Cb[k]))) * dt + self.Cb[k]
        self.i = i_new
        self.Cb = Cb_new
        self.T[:] = self.coolant.T_h(self.i)
        self.rho[:] = self.coolant.rho(self.T)
        return self.i, self.Cb
    
    def get_friction_loss(self, m_dot):
        if abs(m_dot) < 1e-5:
            return 0.0
        mu = self.coolant.my(float(np.mean(self.T)))
        G = np.abs(m_dot) / self.A
        Re = Reynolds(G, self.Dh, mu)
        fd = 4.0 * Colebrook(self.roughness, self.Dh, Re)
        return float(np.sum(fd * self.dx / (2.0 * self.Dh * (self.A**2) * self.rho)))

    # Compute acceleration losses
    def get_acc_loss(self):
        y = 1 / (self.A[-1]**2 * self.rho[-1]) - 1 / (self.A[0]**2 * self.rho[0])
        return y

    def get_Jq(self):
        y = self.rho * np.cos(self.theta)
        y = y[np.abs(y) > 1.0] #makes the integration more accurate removing horizontal parts of the channel
        return np.sum(y) * self.dx