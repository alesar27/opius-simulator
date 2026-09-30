import numpy as np
#Point kinetics
def rho_CR_func(total_time): # controllable control rod reactivity, in pcm
    return -10000 * (total_time > 5) * 0

def point_kinetics_linear(t, y, Delta_Cb, alpha_boron, Delta_temp_moderator, alpha_moderator, Delta_temp_fuel, alpha_fuel_doppler, beta, Lambda, beta_i, lambda_i):
    # Unpack state
    n = y[0]
    C = y[1:] 
    a_B = alpha_boron # boron coefficient in pcm/ppm
    a_M = alpha_moderator # moderator temperature coefficient in pcm/C°
    a_F = alpha_fuel_doppler # fuel temperature coefficient in pcm/C°
    rho_B = a_B * 1e-5 * Delta_Cb
    rho_M = a_M * 1e-5 * Delta_temp_moderator # moderator temperature coefficient , rho a fraction
    rho_F = a_F * 1e-5 * Delta_temp_fuel # fuel temperature coefficient , rho a fraction
    rho_CR = rho_CR_func(t) * 1e-5 # control rod reactivity (for piloting the reactor)
    total_reactivity = rho_B + rho_M + rho_F + rho_CR
    # Derivatives
    dn_dt = ((total_reactivity - beta) / Lambda) * n + np.sum(lambda_i * C)
    dC_dt = (beta_i / Lambda) * n - (lambda_i * C)
    return np.concatenate(([dn_dt], dC_dt))

def p(T_M, C_B, coeffs):
    # 3-rd degree polynomial interpolation, returns the non-normalized reactivity contribution of the moderator and the boron in pcm
    z = 0
    coeff_idx = 0
    for m in range(4):
        for n in range(4 - m):
            z += coeffs[coeff_idx] * (T_M**m) * (C_B**n)
            coeff_idx += 1
    return z

def point_kinetics_nonlinear(t, y, C_B, T_M, MB_coeffs, MB_C, Delta_temp_fuel, alpha_fuel_doppler, beta, Lambda, beta_i, lambda_i):
    # Unpack state
    n = y[0]
    C = y[1:] 
    a_F = alpha_fuel_doppler # fuel temperature coefficient in pcm/C°
    rho_MB = (p(T_M, C_B, MB_coeffs) + MB_C) * 1e-5
    rho_F = a_F * 1e-5 * Delta_temp_fuel # fuel temperature coefficient , rho a fraction
    rho_CR = rho_CR_func(t) * 1e-5 # control rod reactivity (for piloting the reactor)
    total_reactivity = rho_MB + rho_F + rho_CR
    # Derivatives
    dn_dt = ((total_reactivity - beta) / Lambda) * n + np.sum(lambda_i * C)
    dC_dt = (beta_i / Lambda) * n - (lambda_i * C)
    return np.concatenate(([dn_dt], dC_dt))
