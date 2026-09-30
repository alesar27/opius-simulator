import numpy as np

def Dittus_Boetler_h(local_temperature, mb_0, coolant, Area, D_h):
    # Dittus Boetler
    # local coolant properties, at initial pressure and liquid bulk temperature (approximation, could use mean of liquid bulk and outer cladding)
    rho_loc = coolant.rho(local_temperature) # local density
    c_p_loc = coolant.c_p(local_temperature) * 1e3 # local specific heat in SI units
    k_loc = coolant.k(local_temperature) # local thermal conductivity
    my_loc = coolant.my(local_temperature)  # local viscosity
    G_loc = mb_0 / Area # mass flux
    Re_loc = G_loc * D_h / my_loc # local Reynolds
    Pr_loc = c_p_loc * my_loc / k_loc # local Prandelt
    Nu_loc = 0.023 * Re_loc**0.8 * Pr_loc**0.4  # Dittus Boetler (for heating => power of 0.4), local Nusselt
    h_loc = k_loc * Nu_loc / D_h # local heat transfer coefficient with hydraulic diameter as characteristic length
    if Re_loc < 1e4:
        print(f"Re outside of Dittus Boetler validity range")
    if Pr_loc < 0.7 or Pr_loc > 100:
        print(f"Pr outside of Dittus Boetler validity range")
    return h_loc


def HB40_h_correlations(local_temperature, mb_0, coolant, Area, D_h, irradiated):
    # Jeffries harrison correlation for irradiated HB-40
    # Rogers and barns correlation for new unirradiated HB-40
    # local coolant properties, at initial pressure and liquid bulk temperature (approximation, could use mean of liquid bulk and outer cladding)
    c_p_loc = coolant.c_p(local_temperature) * 1e3 # local specific heat in SI units
    k_loc = coolant.k(local_temperature) # local thermal conductivity
    my_loc = coolant.my(local_temperature)  # local viscosity
    #print(my_loc)
    G_loc = mb_0 / Area # mass flux
    Re_loc = G_loc * D_h / my_loc # local Reynolds
    Pr_loc = c_p_loc * my_loc / k_loc # local Prandelt
    if irradiated == True:
        #Jeffries harrison correlation for irradiated HB-40
        Nu_loc = 0.0224 * Re_loc**0.83 * Pr_loc**0.4 
    else:
        # Rogers and barns correlation for new unirradiated HB-40
        Nu_loc = 0.00835 * Re_loc**0.9 * Pr_loc**0.4 
    h_loc = k_loc * Nu_loc / D_h # local heat transfer coefficient with hydraulic diameter as characteristic length
    if Re_loc < 5e3 or Re_loc > 5.5e5:
        print(f"Re outside of correlations validity range = {Re_loc}")
    if Pr_loc < 0.7 or Pr_loc > 100:
        print(f"Pr outside of correlations validity range")
    return h_loc


def CHF_HB40(T_bulk, Pressure, massflow, coolant, Area, irradiated):
    T_sat = np.sqrt((np.log(Pressure) - 0.9454) / 5.568e-6) #P in bar and T_sat in C°
    T_sub = T_sat - T_bulk
    density = coolant.rho(T_bulk)
    velocity = massflow / (Area * density)
    sigma = 362.0 # [kW/(m^2K)]
    if irradiated:
        CHF = 3000 + 7.5 * T_sub ** 0.8 * velocity ** 1.2 # [kW/(m^2K)]
        CHF = CHF - 0.3 * CHF #AECL reccommendation
    else:
        CHF  = 315 + 5.1 * T_sub * velocity ** (2/3) # [kW/(m^2K)]
        CHF = CHF - 2 * sigma #AECL reccommendation
    return CHF

def CHF_Reddyfighetti(x, mass_flux, p, q2):
    x_in = x[0]
    G_r = mass_flux / 1356.23
    p_cr = 220e5
    p = p * 1e5
    p_r = p / p_cr
    q_r = q2 / 3.1544e6
    q_r[0] = 1e-6
    a1 = 0.5328
    a2 = 0.1212
    a3 = -0.3040
    a4 = 0.3285
    c1 = 1.6151
    c2 = 1.4066
    c3 = 0.4843
    c4 = -2.0749
    A = a1 * p_r ** a2 * G_r ** (a3 + a4 * p_r)
    B = 3.1544e6
    C = c1 * p_r ** c2 * G_r ** (c3 + c4 * p_r)
    q_cr = B * (A - x_in) / (C + (x - x_in) / (q_r))
    return q_cr

def Colebrook(roughness, diameter, Reynolds):
    y = (roughness / diameter * 1/3.7)**1.11 + 6.9 / Reynolds
    if np.any(Reynolds) <= 0 : print(f'y error= {Reynolds}')
    LHS = -3.6 * np.log10(y)
    return (1 / LHS)**2

def Reynolds(G, diameter, viscosity):
    return G * diameter / viscosity

# Compute static head integral
def get_Jq(rho, theta, dx):
    y = rho * np.cos(theta)
    y = y[np.abs(y) > 1.0] #makes the integration more accurate removing horizontal parts of the channel
    return np.sum(y) * dx

# Compute inertia
def get_Iq(A, dx):
    I = (np.sum(np.diff(dx / A)))
    return I