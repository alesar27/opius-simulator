import matplotlib.pyplot as plt
import numpy as np

from opius.config.system_configuration import (
    g,
    L_a,
    L_Udl,
    L_Ldl,
    Q_core_SS,
    pump_trip_transient,
    feedwater_loss_transient,
    nonlinear_PK,
    SS_boron_concentration,
    alpha_B,
    alpha_M,
    evaluate_hot,
)
from opius.components.hydraulics_helper import pump_head
from opius.physics.point_kinetics import rho_CR_func, p


def plot_transient(history: dict, sim):
    t = history["t"]
    nt = len(t)
    coolant = sim.HB_conc

    # 1. Flow Rates
    plt.figure(figsize=(10, 6))
    plt.plot(t, history["ma"], label="$m_a(t)$", linewidth=2)
    plt.plot(t, history["mb"], label="$m_b(t)$", linewidth=2)
    plt.plot(t, history["mc"], label="$m_c(t)$", linewidth=2)
    plt.xlabel("Time (s)")
    plt.ylabel("Flow Rate (kg/s)")
    plt.title("O-PIUS Flow Rates")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()

    # 2. Static Head Differences
    plt.figure(figsize=(10, 6))
    plt.plot(t, (history["Ja"] + history["Jb"]) * g, label="$J_a + J_b$", linewidth=2)
    plt.plot(t, (history["Jb"] + history["Jc"]) * g, label="$J_b + J_c$", linewidth=2)
    plt.plot(t, (history["Ja"] - history["Jc"]) * g, label="$J_a - J_c$", linewidth=2)
    plt.xlabel("Time (s)")
    plt.ylabel("Pressure (Pa)")
    plt.title("Static Head Differences")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()

    # 3. Static Head Components & Pump Head
    plt.figure(figsize=(10, 6))
    plt.plot(t, history["Ja"] * g, label="$J_a$", linewidth=2)

    # Dynamic pump head calculation evaluated across all time steps
    i_c_pump = history["i_c"][:, sim.n_pump]
    rho_c_pump = coolant.rho(coolant.T_h(i_c_pump))
    hp_eval = np.array([
        pump_head(ti, sim.h_pump_SS, pump_trip_transient, feedwater_loss_transient)
        for ti in t
    ])
    plt.plot(t, hp_eval * rho_c_pump * g + history["Jc"] * g, label=r"$h_{pump} \rho_0 + J_c$", linewidth=2)
    plt.plot(t, -history["Jb"] * g, label="$-J_b$", linewidth=2)
    plt.xlabel("Time (s)")
    plt.ylabel("Static Heads (Pa)")
    plt.title("Static Head with Pump Trip $h_p(t)$")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()

    # 4. Channel A Axial Enthalpy Evolution
    plt.figure(figsize=(10, 6))
    plt.plot(sim.x_a, history["i_a"][0, :], label="$i_a(0)$", linewidth=2)
    plt.plot(sim.x_a, history["i_a"][int(nt / 4), :], label="$i_a(1/4)$", linewidth=2)
    plt.plot(sim.x_a, history["i_a"][int(nt / 2), :], label="$i_a(1/2)$", linewidth=2)
    plt.plot(sim.x_a, history["i_a"][-1, :], label="$i_a(end)$", linewidth=2)
    plt.axvspan(0, L_Udl, color="gray", alpha=0.2, label="Density Lock")
    plt.axvspan(L_a - L_Ldl, L_a, color="gray", alpha=0.2)
    plt.xlabel("Distance (m)")
    plt.ylabel("Enthalpy (kJ/kg)")
    plt.title("Channel A Enthalpy Evolution")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()

    # 5. Channel B Axial Enthalpy Evolution
    plt.figure(figsize=(10, 6))
    plt.plot(sim.x_b, history["i_b"][0, :], label="$i_b(0)$", linewidth=2)
    plt.plot(sim.x_b, history["i_b"][int(nt / 4), :], label="$i_b(1/4)$", linewidth=2)
    plt.plot(sim.x_b, history["i_b"][int(nt / 2), :], label="$i_b(1/2)$", linewidth=2)
    plt.plot(sim.x_b, history["i_b"][-1, :], label="$i_b(end)$", linewidth=2)
    plt.xlabel("Distance (m)")
    plt.ylabel("Enthalpy (kJ/kg)")
    plt.title("Channel B Enthalpy Evolution")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()

    # 6. Channel B Axial Temperature Evolution
    plt.figure(figsize=(10, 6))
    plt.plot(sim.x_b, history["T_b"][0, :], label="$T_b(0)$", linewidth=2)
    plt.plot(sim.x_b, history["T_b"][int(nt / 4), :], label="$T_b(1/4)$", linewidth=2)
    plt.plot(sim.x_b, history["T_b"][int(nt / 2), :], label="$T_b(1/2)$", linewidth=2)
    plt.plot(sim.x_b, history["T_b"][-1, :], label="$T_b(end)$", linewidth=2)
    plt.xlabel("Distance (m)")
    plt.ylabel("Temperature (°C)")
    plt.title("Channel B Temperature Evolution")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()

    # 7. Channel C Axial Enthalpy Evolution
    plt.figure(figsize=(10, 6))
    plt.plot(sim.x_c, history["i_c"][0, :], label="$i_c(0)$", linewidth=2)
    plt.plot(sim.x_c, history["i_c"][int(nt / 4), :], label="$i_c(1/4)$", linewidth=2)
    plt.plot(sim.x_c, history["i_c"][int(nt / 2), :], label="$i_c(1/2)$", linewidth=2)
    plt.plot(sim.x_c, history["i_c"][-1, :], label="$i_c(end)$", linewidth=2)
    plt.xlabel("Distance (m)")
    plt.ylabel("Enthalpy (kJ/kg)")
    plt.title("Channel C Enthalpy Evolution")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()

    # 8. Core Outlet Temperatures
    plt.figure(figsize=(10, 6))
    plt.plot(t, history["T_b"][:, sim.idx_exit_core], label="$T_b$ (Average)", linewidth=2)
    if evaluate_hot and "T_b_hot" in history:
        plt.plot(t, history["T_b_hot"][:, sim.idx_exit_core], label="$T_{b-hot}$", linewidth=2, linestyle="--")
    plt.xlabel("Time (s)")
    plt.ylabel("Temperature (°C)")
    plt.title("Core Outlet Temperature")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()

    # 9. Steam Generator Outlet Temperature
    plt.figure(figsize=(10, 6))
    plt.plot(t, coolant.T_h(history["i_c"][:, sim.idx_exit_SG]), linewidth=2)
    plt.xlabel("Time (s)")
    plt.ylabel("Temperature (°C)")
    plt.title("Steam Generator Outlet Temperature")
    plt.grid(True, alpha=0.3)
    plt.show()

    # 10. Channel A Boron Axial Evolution
    plt.figure(figsize=(10, 6))
    plt.plot(sim.x_a, history["Cb_a"][0, :], label="$Cb_a(0)$", linewidth=2)
    plt.plot(sim.x_a, history["Cb_a"][int(nt / 4), :], label="$Cb_a(1/4)$", linewidth=2)
    plt.plot(sim.x_a, history["Cb_a"][int(nt / 2), :], label="$Cb_a(1/2)$", linewidth=2)
    plt.plot(sim.x_a, history["Cb_a"][-1, :], label="$Cb_a(end)$", linewidth=2)
    plt.axvspan(0, L_Udl, color="gray", alpha=0.2, label="Density Lock")
    plt.axvspan(L_a - L_Ldl, L_a, color="gray", alpha=0.2)
    plt.xlabel("Distance (m)")
    plt.ylabel("Boron Concentration (ppm)")
    plt.title("Channel A Boron Concentration")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()

    # 11. Channel B Boron Axial Evolution
    plt.figure(figsize=(10, 6))
    plt.plot(sim.x_b, history["Cb_b"][0, :], label="$Cb_b(0)$", linewidth=2)
    plt.plot(sim.x_b, history["Cb_b"][int(nt / 4), :], label="$Cb_b(1/4)$", linewidth=2)
    plt.plot(sim.x_b, history["Cb_b"][int(nt / 2), :], label="$Cb_b(1/2)$", linewidth=2)
    plt.plot(sim.x_b, history["Cb_b"][-1, :], label="$Cb_b(end)$", linewidth=2)
    plt.xlabel("Distance (m)")
    plt.ylabel("Boron Concentration (ppm)")
    plt.title("Channel B Boron Concentration")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()

    # 12. Core Inlet Boron
    plt.figure(figsize=(10, 6))
    plt.plot(t, history["Cb_b"][:, sim.idx_start_core], linewidth=2)
    plt.xlabel("Time (s)")
    plt.ylabel("Boron Concentration (ppm)")
    plt.title("Boron Concentration at Core Inlet")
    plt.grid(True, alpha=0.3)
    plt.show()

    # 13. Relative Core Power
    plt.figure(figsize=(10, 6))
    plt.plot(t, history["Q_core"] / Q_core_SS, linewidth=2)
    plt.xlabel("Time (s)")
    plt.ylabel("Relative Power (-)")
    plt.title("Relative Core Power")
    plt.grid(True, alpha=0.3)
    plt.show()

    # 14. Absolute Power Comparison
    plt.figure(figsize=(10, 6))
    plt.plot(t, np.abs(history["Q_core"]), label="Core Power", linewidth=2)
    plt.plot(t, np.abs(history["P_b"]), label="Power to Coolant in B", linewidth=2)
    plt.plot(t, np.abs(history["P_c"]), label="Power from Coolant in C", linewidth=2)
    plt.xlabel("Time (s)")
    plt.ylabel("Power (kW)")
    plt.title("Core Power Evolution (Absolute Values)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()

    # 15. Reactivity Feedback Breakdown
    plt.figure(figsize=(10, 6))
    if nonlinear_PK:
        total_mod_boron = p(
            history["T_avg_core"],
            history["Cb_avg"] + SS_boron_concentration,
            sim.MB_3rd_degree_polynomial_interpolation_coeffs,
        ) + sim.MB_normalization_constant
        total_doppler = (history["T_avg_fuel"] - history["T_avg_fuel"][0]) * sim.alpha_doppler
        total_cr = rho_CR_func(t)
        total_rho = total_mod_boron + total_doppler + total_cr

        plt.plot(t, total_mod_boron, label="Moderator and Boron (pcm)", linewidth=2)
        plt.plot(t, total_doppler, label="Doppler Feedback (pcm)", linewidth=2)
        plt.plot(t, total_cr, label="Control Rods (pcm)", linewidth=2)
        plt.plot(t, total_rho, label="Total Reactivity (pcm)", linewidth=2)
    else:
        total_boron = (history["Cb_avg"] - history["Cb_avg"][0]) * alpha_B
        total_mod = (history["T_avg_core"] - history["T_avg_core"][0]) * alpha_M
        total_doppler = (history["T_avg_fuel"] - history["T_avg_fuel"][0]) * sim.alpha_doppler
        total_cr = rho_CR_func(t)
        total_rho = total_boron + total_mod + total_doppler + total_cr

        plt.plot(t, total_boron, label="Boron (pcm)", linewidth=2)
        plt.plot(t, total_mod, label="Moderator (pcm)", linewidth=2)
        plt.plot(t, total_doppler, label="Doppler Feedback (pcm)", linewidth=2)
        plt.plot(t, total_cr, label="Control Rods (pcm)", linewidth=2)
        plt.plot(t, total_rho, label="Total Reactivity (pcm)", linewidth=2)

    plt.xlabel("Time (s)")
    plt.ylabel("Reactivity (pcm)")
    plt.title("Reactivity Contributions")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()

    # Thermal Margins & Fuel Conduction Diagnostics#

    # 1. MDNBR Plot (evaluated against time)
    if sim.evaluate_CHF and "MDNBR" in history and len(history["MDNBR"]) > 0:
        plt.figure(figsize=(10, 6))
        plt.plot(t, history["MDNBR"], label="$MDNBR$", linewidth=2, color="crimson")
        plt.axhline(1.0, linestyle="--", label="DNB Limit (1.0)")
        plt.xlabel("Time [s]")
        plt.ylabel("MDNBR [-]")
        plt.ylim(bottom=1.0, top=15.0)
        plt.title("Minimum DNB Ratio (MDNBR)")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.show()

    # 2. Extract active fuel pin nodes from the driver instance
    fuel_nodes = [pin for pin in sim.core if pin is not None]
    hot_fuel_nodes = [pin for pin in sim.core_hot if pin is not None] if evaluate_hot else []

    if len(fuel_nodes) > 0:
        # Axial distribution of average pin centerline temperature
        zs = [node.z for node in fuel_nodes]
        center_T_init = [node.T_centerline[0] for node in fuel_nodes]
        center_T_quarter = [node.T_centerline[int(nt / 4)] for node in fuel_nodes]
        center_T_middle = [node.T_centerline[int(nt / 2)] for node in fuel_nodes]
        center_T_end = [node.T_centerline[-1] for node in fuel_nodes]

        plt.figure(figsize=(10, 6))
        plt.plot(zs, center_T_init, label=r"$T_{center}(0)$", linewidth=2)
        plt.plot(zs, center_T_quarter, label=r"$T_{center}(1/4)$", linewidth=2)
        plt.plot(zs, center_T_middle, label=r"$T_{center}(1/2)$", linewidth=2)
        plt.plot(zs, center_T_end, label=r"$T_{center}(1/1)$", linewidth=2)
        plt.xlabel("Core Height (m)")
        plt.ylabel("Average pin fuel centerline temperature (°C)")
        plt.title("Axial distribution of average pin fuel centerline temperature")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.show()

        # Axial distribution of average pin outer cladding / film temperature
        co_T_init = [node.T[0, node.N] for node in fuel_nodes]
        co_T_quarter = [node.T[int(nt / 4), node.N] for node in fuel_nodes]
        co_T_middle = [node.T[int(nt / 2), node.N] for node in fuel_nodes]
        co_T_end = [node.T[-1, node.N] for node in fuel_nodes]

        plt.figure(figsize=(10, 6))
        plt.plot(zs, co_T_init, label=r"$T_{co}(0)$", linewidth=2)
        plt.plot(zs, co_T_quarter, label=r"$T_{co}(1/4)$", linewidth=2)
        plt.plot(zs, co_T_middle, label=r"$T_{co}(1/2)$", linewidth=2)
        plt.plot(zs, co_T_end, label=r"$T_{co}(1/1)$", linewidth=2)
        plt.xlabel("Core Height (m)")
        plt.ylabel("Average pin outer cladding / film temperature (°C)")
        plt.title("Axial distribution of average pin outer cladding / film temperature")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.show()

    # 3. Hot Pin Axial Profiles
    if evaluate_hot and len(hot_fuel_nodes) > 0:
        # Axial distribution of hot pin centerline temperature
        zs_hot = [node.z for node in hot_fuel_nodes]
        center_T_init_hot = [node.T_centerline[0] for node in hot_fuel_nodes]
        center_T_quarter_hot = [node.T_centerline[int(nt / 4)] for node in hot_fuel_nodes]
        center_T_middle_hot = [node.T_centerline[int(nt / 2)] for node in hot_fuel_nodes]
        center_T_end_hot = [node.T_centerline[-1] for node in hot_fuel_nodes]

        plt.figure(figsize=(10, 6))
        plt.plot(zs_hot, center_T_init_hot, label=r"$T_{center}(0)$ Hot pin", linewidth=2)
        plt.plot(zs_hot, center_T_quarter_hot, label=r"$T_{center}(1/4)$ Hot pin", linewidth=2)
        plt.plot(zs_hot, center_T_middle_hot, label=r"$T_{center}(1/2)$ Hot pin", linewidth=2)
        plt.plot(zs_hot, center_T_end_hot, label=r"$T_{center}(1/1)$ Hot pin", linewidth=2)
        plt.xlabel("Core Height (m)")
        plt.ylabel("Hot pin fuel centerline temperature (°C)")
        plt.title("Axial distribution of hot pin fuel centerline temperature")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.show()

        # Axial distribution of hot pin outer cladding / film temperature
        co_T_init_hot = [node.T[0, node.N] for node in hot_fuel_nodes]
        co_T_quarter_hot = [node.T[int(nt / 4), node.N] for node in hot_fuel_nodes]
        co_T_middle_hot = [node.T[int(nt / 2), node.N] for node in hot_fuel_nodes]
        co_T_end_hot = [node.T[-1, node.N] for node in hot_fuel_nodes]

        plt.figure(figsize=(10, 6))
        plt.plot(zs_hot, co_T_init_hot, label=r"$T_{co}(0)$ Hot pin", linewidth=2)
        plt.plot(zs_hot, co_T_quarter_hot, label=r"$T_{co}(1/4)$ Hot pin", linewidth=2)
        plt.plot(zs_hot, co_T_middle_hot, label=r"$T_{co}(1/2)$ Hot pin", linewidth=2)
        plt.plot(zs_hot, co_T_end_hot, label=r"$T_{co}(1/1)$ Hot pin", linewidth=2)
        plt.xlabel("Core Height (m)")
        plt.ylabel("Hot pin outer cladding / film temperature (°C)")
        plt.title("Axial distribution of hot pin outer cladding / film temperature")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.show()