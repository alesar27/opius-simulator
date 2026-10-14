# SS Coolant and Cost Model
from Economics.coolantProperties import *
import matplotlib.pyplot as plt
import os

plt.rcParams.update({
    'axes.titlesize': 16,   # Title
    'axes.labelsize': 16,   # X and Y labels
    'xtick.labelsize': 13,  # X tick labels
    'ytick.labelsize': 13,  # Y tick labels
    'legend.fontsize': 14,  # Legend
})

FOLDER = ""

# Core Parameters
Power = 2e9 # [W], core power, approx fission power / 1.05
L_hot = 25 # [m], hot length
L_cold = 125 # [m], cold length
f_FP = 0.015 # [-], fraction of fission power deposited in coolant
HBs = np.array([10, 20, 30]) # [%], steady-state HB fraction
VOL = 5 # [%], steady-state VOL fraction in the "worst" case (no recycling)
eta_distil = 0.9 # [-], distillation efficiency

# Operating Parameters (steady state)
T_hots = np.arange(320, 410, 5) # [°C], outlet temperature
Delta_T = 60 # [°C], through-core temperature increase
v_hot = 8 # [m/s], hot coolant speed
v_cold = 8 # [m/s], cold coolant speed

# Economic parameters
fresh_coolant_cost_kg = 7.2 # [€/kg]
electricity_price_kWhe = 0.1 # [€/kWhe]


# radiolytic decomposition rate
def D_R(coolant, f_FP, FP):
    # FP in kW, returns kg/h
    return 13.1e-3 * f_FP * FP * ((100 - coolant.HB) / 70)**2

# pyrolytic decomposition rate, per temperature zone (to analyze hot and cold zone contributions)
def D_P(coolant, T, v, L, mass_flow_rate):
    # returns kg/h
    T_K = T + 273.15 # input temperature in °C but formula for K
    W = mass_flow_rate / v * L # weight of coolant in the region
    return (100 - coolant.HB) / 70 * W * np.exp(25.73 - 180.98 / (8.314e-3 * T_K)) # R in kJ/(mol K), so 8.314e-3 instead of the SI 8.314

# decomposition products fractions
def fracDecProd(T, product='HB'):
    if product=='HB':
        A, B, C, D = 3.095, -1.107e-2, 0.909e-5, 6e-3
    elif product=='VOL':
        A, B, C, D = -3.479, 2.069e-2, -2.595e-5, -6e-3
    elif product=='GAS':
        A, B, C, D = 1.384, -0.962e-2, 1.686e-5, 0
    else:
        raise "Wrong Decomposition Product Input"
    return A + B * T + C * T**2 + D * HB

# net efficiency as a function of outler temperature
def eta_net_func(T_hot, T_ambient=20):
    return 1 - ((T_ambient + 273.15) / (T_hot + 273.15))**0.58

# net efficiency plot at T_ambient=20¨C
Ts = np.arange(280.0, 450.0, 5)
print(Ts)
etas = eta_net_func(Ts)
plt.figure(figsize=(10, 6))
plt.plot(Ts, etas * 100)
plt.plot(291, 31, 'go',label="PIUS")
plt.plot(327, 34, 'ro',label="AP1000")
plt.plot(330, 36, 'bo',label="EPR")
plt.xlabel("Core Outlet Temperature [°C]")
plt.ylabel(f"Plant Net Efficiency [%]")
plt.grid(True, alpha=0.3)
plt.legend()
plt.savefig(os.path.join(FOLDER, "eta_net"))
plt.show()


# ========= Calculations =====================================
difference_eletricity_return_makeup_cost_array = []
makeup_cost_per_kWhe_array_no_recycling = np.zeros((len(HBs) , len(T_hots)))
makeup_cost_per_kWhe_array_with_recycling = np.zeros((len(HBs) , len(T_hots)))
HB_production_rate_array = np.zeros((len(HBs) , len(T_hots)))
VOL_production_rate_array = np.zeros((len(HBs) , len(T_hots)))
GAS_production_rate_array = np.zeros((len(HBs) , len(T_hots)))
makeup_rate_FB_array = np.zeros((len(HBs), len(T_hots)))
fraction_radio_decomp_array = np.zeros((len(HBs), len(T_hots)))
fraction_pyro_decomp_hot_array = np.zeros((len(HBs), len(T_hots)))
fraction_pyro_decomp_cold_array = np.zeros((len(HBs), len(T_hots)))

for i, HB in enumerate(HBs):
    for j, T_hot in enumerate(T_hots):
        T_cold = T_hot - Delta_T
        print(f"HB = {HB} % , T_hot = {T_hot}")
        coolant = HB40(HB)

        mass_flow_rate = Power / coolant.c_p_int(T_cold, T_hot)
        print(f"Mass flow rate = {mass_flow_rate} kg/s")

        FP = Power * 1e-3 * 1.05 # Fission power in kW
        radio_decomp_rate = D_R(coolant, f_FP, FP)
        print(f"D_R = {radio_decomp_rate} kg/h")

        pyro_decomp_rate_hot = D_P(coolant, T_hot, v_hot, L_hot, mass_flow_rate)
        pyro_decomp_rate_cold = D_P(coolant, T_cold, v_cold, L_cold, mass_flow_rate)
        print(f"D_P_hot = {pyro_decomp_rate_hot} kg/h")
        print(f"D_P_cold = {pyro_decomp_rate_cold} kg/h")

        A_hot = mass_flow_rate / v_hot / coolant.rho(T_hot)
        A_cold = mass_flow_rate / v_cold / coolant.rho(T_cold)
        print(f"A_hot = {A_hot} m^2")
        print(f"A_cold = {A_cold} m^2")

        D_TOT = radio_decomp_rate + pyro_decomp_rate_hot + pyro_decomp_rate_cold
        print(f"D_TOT = {D_TOT} kg/h")

        fraction_radio_decomp = radio_decomp_rate / D_TOT
        fraction_radio_decomp_array[i][j] = fraction_radio_decomp
        fraction_pyro_decomp_hot = pyro_decomp_rate_hot / D_TOT
        fraction_pyro_decomp_hot_array[i][j] = fraction_pyro_decomp_hot
        fraction_pyro_decomp_cold = pyro_decomp_rate_cold / D_TOT
        fraction_pyro_decomp_cold_array[i][j] = fraction_pyro_decomp_cold

        print(f"fraction_radio_decomp = {fraction_radio_decomp * 100} %")
        print(f"fraction_pyro_decomp_hot = {fraction_pyro_decomp_hot * 100} %")
        print(f"fraction_pyro_decomp_cold = {fraction_pyro_decomp_cold * 100} %")

        # feed and bleed based on HB removal
        f_HB_core = fracDecProd((T_hot+T_cold)/2, 'HB')
        f_HB_hot = fracDecProd(T_hot, 'HB')
        f_HB_cold = fracDecProd(T_cold, 'HB')
        print(f"f_HB_core = {f_HB_core} , f_HB_hot = {f_HB_hot} , f_HB_cold = {f_HB_cold}")
        f_VOL_core = fracDecProd((T_hot+T_cold)/2, 'VOL')
        f_VOL_hot = fracDecProd(T_hot, 'VOL')
        f_VOL_cold = fracDecProd(T_cold, 'VOL')
        print(f"f_VOL_core = {f_VOL_core} , f_VOL_hot = {f_VOL_hot} , f_VOL_cold = {f_VOL_cold}")
        f_GAS_core = fracDecProd((T_hot+T_cold)/2, 'GAS')
        f_GAS_hot = fracDecProd(T_hot, 'GAS')
        f_GAS_cold = fracDecProd(T_cold, 'GAS')
        print(f"f_GAS_core = {f_GAS_core} , f_GAS_hot = {f_GAS_hot} , f_GAS_cold = {f_GAS_cold}")

        HB_production_rate = radio_decomp_rate * f_HB_core + pyro_decomp_rate_hot * f_HB_hot + pyro_decomp_rate_cold * f_HB_cold
        HB_production_rate_array[i][j] = HB_production_rate
        VOL_production_rate = radio_decomp_rate * f_VOL_core + pyro_decomp_rate_hot * f_VOL_hot + pyro_decomp_rate_cold * f_VOL_cold
        VOL_production_rate_array[i][j] = VOL_production_rate
        GAS_production_rate = radio_decomp_rate * f_GAS_core + pyro_decomp_rate_hot * f_GAS_hot + pyro_decomp_rate_cold * f_GAS_cold
        GAS_production_rate_array[i][j] = GAS_production_rate
        
        print(f"HB production rate = {HB_production_rate} kg/h")
        print(f"VOL production rate = {VOL_production_rate} kg/h")
        print(f"GAS production rate = {GAS_production_rate} kg/h")

        # No recycling (worst case)
        FB = HB_production_rate / (HB * 1e-2)
        makeup_rate = FB * (1 - VOL * 1e-2) + GAS_production_rate + VOL_production_rate
        print(f"Makeup rate = {makeup_rate} kg/h")

        makeup_cost = makeup_rate * fresh_coolant_cost_kg # [€/h]
        eta_net = eta_net_func(T_hot) # [-]
        electricy_return = Power / 1e3 * eta_net * electricity_price_kWhe # [€/h]
        ratio_makeup_cost_eletricity_return = makeup_cost / electricy_return # [-]
        difference_eletricity_return_makeup_cost = electricy_return - makeup_cost # [€ / h]
        difference_eletricity_return_makeup_cost_array.append(difference_eletricity_return_makeup_cost)
        makeup_cost_per_kWhe = makeup_cost / (Power / 1e3 * eta_net)
        makeup_cost_per_kWhe_array_no_recycling[i][j] = makeup_cost_per_kWhe
        print(f"makeup_cost = {makeup_cost} €/h")
        print(f"makeup_cost_per_kWhe = {makeup_cost_per_kWhe} €/kWhe")
        print(f"net thermal efficiency = {eta_net}")
        print(f"electricity_return = {electricy_return} €/h")
        print(f"ratio between the 2 costs = {ratio_makeup_cost_eletricity_return * 100} %")
        print(f"difference between the 2 costs = {difference_eletricity_return_makeup_cost} €/h")
        print("\n=====================================\n")

        # With Recycling (best case)
        makeup_rate = HB_production_rate / eta_distil + GAS_production_rate
        print(f"Makeup rate = {makeup_rate} kg/h")

        makeup_cost = makeup_rate * fresh_coolant_cost_kg # [€/h]
        eta_net = eta_net_func(T_hot) # [-]
        electricy_return = Power / 1e3 * eta_net * electricity_price_kWhe # [€/h]
        ratio_makeup_cost_eletricity_return = makeup_cost / electricy_return # [-]
        difference_eletricity_return_makeup_cost = electricy_return - makeup_cost # [€ / h]
        difference_eletricity_return_makeup_cost_array.append(difference_eletricity_return_makeup_cost)
        makeup_cost_per_kWhe = makeup_cost / (Power / 1e3 * eta_net)
        makeup_cost_per_kWhe_array_with_recycling[i][j] = makeup_cost_per_kWhe
        print(f"makeup_cost = {makeup_cost} €/h")
        print(f"makeup_cost_per_kWhe = {makeup_cost_per_kWhe} €/kWhe")
        print(f"net thermal efficiency = {eta_net}")
        print(f"electricity_return = {electricy_return} €/h")
        print(f"ratio between the 2 costs = {ratio_makeup_cost_eletricity_return * 100} %")
        print(f"difference between the 2 costs = {difference_eletricity_return_makeup_cost} €/h")
        print("\n=====================================\n")
# Title, reflects the conditions
# title = f"P={int(Power/1e9)}GW, f_FP={f_FP}%, L_hot={L_hot}m, L_cold={L_cold}m, v_hot={v_hot}m/s, v_cold={v_cold}m/s, Delta_T={Delta_T}°C"

# make-up cost per kWhe
plt.figure(figsize=(10, 6))
for i in range(len(HBs)):
    plt.plot(T_hots, makeup_cost_per_kWhe_array_no_recycling[i] * 100, label=f"HB : {HBs[i]} [%]", marker="o")
plt.xlabel("Core Outlet Temperature [°C]")
plt.ylabel(r"Coolant Make-Up Cost$\left[\frac{\text{c€}}{\text{kWhe}}\right]$")
plt.grid(True, alpha=0.3)
plt.legend()
plt.savefig(os.path.join(FOLDER, "MC_no_recycling_kWhe"))
plt.show()

plt.figure(figsize=(10, 6))
for i in range(len(HBs)):
    plt.plot(T_hots, makeup_cost_per_kWhe_array_with_recycling[i] * 100, label=f"HB : {HBs[i]} [%]", marker="o")
plt.xlabel("Core Outlet Temperature [°C]")
plt.ylabel(r"Coolant Make-Up Cost$\left[\frac{\text{c€}}{\text{kWhe}}\right]$")
plt.grid(True, alpha=0.3)
plt.legend()
plt.savefig(os.path.join(FOLDER, "MC_with_recycling_kWhe"))
plt.show()

# HB production rate
plt.figure(figsize=(10, 6))
for i in range(len(HBs)):
    plt.plot(T_hots, HB_production_rate_array[i], label=f"HB : {HBs[i]} [%]", marker="o")
plt.xlabel("Core Outlet Temperature [°C]")
plt.ylabel(r"HB Production Rate$\left[\frac{\text{kg}}{\text{h}}\right]$")
plt.grid(True, alpha=0.3)
plt.legend()
plt.savefig(os.path.join(FOLDER, "HB_prod_rate"))
plt.show()

# VOL production rate
plt.figure(figsize=(10, 6))
for i in range(len(HBs)):
    plt.plot(T_hots, VOL_production_rate_array[i], label=f"HB : {HBs[i]} [%]", marker="o")
plt.xlabel("Core Outlet Temperature [°C]")
plt.ylabel(r"VOL Production Rate$\left[\frac{\text{kg}}{\text{h}}\right]$")
plt.grid(True, alpha=0.3)
plt.legend()
plt.savefig(os.path.join(FOLDER, "VOL_prod_rate"))
plt.show()

# GAS production rate
plt.figure(figsize=(10, 6))
for i in range(len(HBs)):
    plt.plot(T_hots, GAS_production_rate_array[i], label=f"HB : {HBs[i]} [%]", marker="o")
plt.xlabel("Core Outlet Temperature [°C]")
plt.ylabel(r"GAS Production Rate$\left[\frac{\text{kg}}{\text{h}}\right]$")
plt.grid(True, alpha=0.3)
plt.legend()
plt.savefig(os.path.join(FOLDER, "GAS_prod_rate"))
plt.show()

# Decomposition rate fraction
plt.figure(figsize=(10, 6))
for i in range(len(HBs)):
    plt.plot(T_hots, fraction_radio_decomp_array[i] * 100, label=f"Radiolytic Decomposition (HB : {HBs[i]} [%])", marker="o")
    plt.plot(T_hots, (fraction_pyro_decomp_hot_array[i] + fraction_pyro_decomp_cold_array[i]) * 100, label=f"Pyrolitic Decomposition (HB : {HBs[i]} [%])", marker="o")

plt.xlabel("Core Outlet Temperature [°C]")
plt.ylabel(r"Decomposition Rate Fractions [%]")
plt.grid(True, alpha=0.3)
plt.legend()
plt.savefig(os.path.join(FOLDER, "decomposition_rate_fractions"))
plt.show()