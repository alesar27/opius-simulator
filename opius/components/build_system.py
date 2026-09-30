import numpy as np
from opius.config.system_configuration import *
from opius.components.channel_1D import Channel1D

def build_channels(coolant):
    # Channel A: Pool
    x_a = np.linspace(0, L_a, nx_a)
    theta_a = np.where(x_a >= (L_a - L_lower_plenum), np.pi, 0.0)
    channel_a = Channel1D("Pool_A", L_a, nx_a, A_a_in, theta_a, roughness_else, coolant)

    n_Udl = int(L_Udl / channel_a.dx)
    n_Ldl = int(L_Ldl / channel_a.dx)
    A_LDl = (id_tube * 1e-3 * np.pi / 4.0)**2 * n_tubes_LDl
    A_UDl = (id_tube * 1e-3 * np.pi / 4.0)**2 * n_tubes_UDl
    L_interface_U = L_Udl * 0.5  # m, position of the interface between primary loop and pool from beginning of Ch. A relative to density lock length
    L_interface_L = L_a - L_Ldl * 0.5
    dx_a = L_a/ (nx_a - 1)
    n_interface_Ldl = int((L_a - L_interface_L) / dx_a)
    if n_interface_Ldl == 0 :
        n_interface_Ldl = 1
    n_interface_Udl = int(L_interface_U/ dx_a)
    if n_interface_Udl == 0 :
        n_interface_Udl = 1
    channel_a.A[0:n_Udl] = A_UDl
    channel_a.A[-n_Ldl:] = A_LDl

    # Channel B: Core & Riser
    x_b = np.linspace(0, L_b, nx_b)
    idx_start_core = np.abs(x_b - x_start_core).argmin()
    idx_exit_core = np.abs(x_b - (x_start_core + H_core)).argmin()
    area_b = np.full(nx_b, A_b)
    area_b[idx_start_core:idx_exit_core] = A_core
    roughness_b = np.full(nx_b, roughness_else)
    roughness_b[idx_start_core:idx_exit_core] = roughness_core
    channel_b = Channel1D("Core_B", L_b, nx_b, area_b, np.pi, roughness_b, coolant)
    channel_b.Dh[idx_start_core:idx_exit_core] = 4.0 * (pitch**2 - np.pi * (d_r/2)**2) / (np.pi * d_r)

    # Channel C: Loop
    x_c = np.linspace(0, L_c, nx_c)
    sector_c = [
        (x_c < 13.25),
        (x_c >= 13.25) & (x_c < 38.25),
        (x_c >= 38.25) & (x_c < 53.25),
        (x_c >= 53.25) & (x_c < 68.25),
        (x_c >= 68.25) & (x_c < 87.25),
        (x_c >= 87.25)
    ]
    angles_c = [np.pi, np.pi / 2.0, 0.0, np.pi, np.pi / 2.0, 0.0]
    areas_c = [A_b, A_c_hl, A_c_SG, A_c_cl, A_c_cl, A_c_downcomer]
    theta_c = np.select(sector_c, angles_c)
    area_c = np.select(sector_c, areas_c)

    channel_c = Channel1D("Loop_C", L_c, nx_c, area_c, theta_c, roughness_else, coolant)

    return channel_a, channel_b, channel_c, (idx_start_core, idx_exit_core), (A_LDl, A_UDl), (n_interface_Ldl, n_interface_Udl)
