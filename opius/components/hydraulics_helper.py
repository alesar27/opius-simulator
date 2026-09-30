import numpy as np
from opius.config.system_configuration import *

def pump_head(total_time, h_pump_SS, pump_trip_transient, feedwater_loss_transient):
    if pump_trip_transient:
        # Speed drops by ~35% in 6 seconds following pump trip
        return h_pump_SS * (0.73 + 0.27 * (1.0 / (1.5 * total_time + 1.0)) ** 2)
    elif feedwater_loss_transient:
        return h_pump_SS
    return h_pump_SS

def derivatives_mq(t, y, hp, ch_a, ch_b, ch_c, F_a, F_b, F_c, rho_pump, g = 9.81):
    # variables
    ma, mb = y
    mc = mb - ma

    #pressure drops
    delta_Pa = F_a * np.abs(ma) * ma 
    delta_Pb = F_b * np.abs(mb) * mb 
    delta_Pc = F_c * np.abs(mc) * mc 

    J_a = ch_a.get_Jq()
    J_b = ch_b.get_Jq()
    J_c = ch_c.get_Jq()

    I_a = ch_a.inertia
    I_b = ch_b.inertia
    I_c = ch_c.inertia
    
    #Define Matrix A 
    A = np.array([
        [I_a,  I_b],
        [-I_c, (I_c + I_b)]
    ])
    
    #Define Vector B 
    B = np.array([
        g * (J_a + J_b) - (delta_Pa + delta_Pb),
        g * (J_b + J_c) - (delta_Pb + delta_Pc) + rho_pump * g * hp
    ])
    # solve for derivatives
    d_dt = np.linalg.solve(A, B)
    return d_dt

def S_b_func(core, n, P_H):
    # returns volumetric power to the coolant in channel b based on the core object array at time n
    toReturn = np.zeros(len(core))
    last = 0
    for i in range(len(core)):
        if core[i] is not None: 
            toReturn[i] = core[i].hf_co[n] * P_H / A_core # volumetric power in the coolant from fuel heat flux
            last=i
        else:
            toReturn[i] = 0
    return toReturn / 1000.0 # Hydraulics uses kW and fuel uses W

def S_a_func(total_time):
    return np.zeros(nx_a)

def S_c_func(total_time, x_c, S_c_correction_factor):
    if pump_trip_transient:
        S_c_transient = np.where((x_c >= x_start_SG) & (x_c <= x_start_SG + H_SG), -Q_core_SS / H_SG * S_c_correction_factor, 0) / A_c_SG * ((total_time <= 0.1) + (total_time >= 0.1) * (total_time <= 18) * (0.75 + 0.25 * (np.exp( - (total_time - 0.1) / 4.3))) +  0.75 * (total_time > 18) * (np.exp( - (total_time - 18) / .2)))
    elif feedwater_loss_transient:
        S_c_transient = np.where((x_c >= x_start_SG) & (x_c <= x_start_SG + H_SG), -Q_core_SS / H_SG * S_c_correction_factor, 0) / A_c_SG * ((total_time <= 0.1) + (total_time >= 0.1) * (total_time <= 35) * (0.75 + 0.25 * 4 * (np.exp( - (total_time - 0.1) / 5))) +  0.75 * (total_time > 35) * (np.exp( - (total_time - 35) / 1.5)))
    else :
        S_c_transient = np.where((x_c >= x_start_SG) & (x_c <= x_start_SG + H_SG), -Q_core_SS / H_SG * S_c_correction_factor, 0) / A_c_SG
    return S_c_transient