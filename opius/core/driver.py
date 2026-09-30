from functools import partial
from pathlib import Path
import numpy as np
import scipy as sp
from scipy.integrate import solve_ivp
from scipy.special import j1

# System Configuration & Data
from opius.config.system_configuration import *

# 2. Physics
from opius.physics.coolantProperties import HB40
from opius.physics.fuel_conduction_Model import Fuel
from opius.physics.physics_helper import HB40_h_correlations, CHF_HB40
from opius.physics.point_kinetics import (point_kinetics_linear, point_kinetics_nonlinear, p,)

# Components and hydraulics
from opius.components.channel_1D import Channel1D
from opius.components.density_locks import solve_density_locks
from opius.components.hydraulics_helper import (pump_head, derivatives_mq, S_a_func, S_b_func, S_c_func,)
from opius.components.build_system import build_channels


class OPIUSDriver:
    def __init__(self, data_dir=None):
        # Resolve data path dynamically
        if data_dir is not None:
            self.data_dir = Path(data_dir).resolve()
        else:
            self.data_dir = (Path(__file__).resolve().parent.parent / "data").resolve()
        # -- Parameters and safegaurds -- #
        if pump_trip_transient and feedwater_loss_transient:
            raise ValueError("Cannot run 2 transients simultaneously")
        self.evaluate_CHF = evaluate_hot and evaluate_CHF

        self.HB_conc = HB40(HB=HB_wt)
        self.irradiated = self.HB_conc.HB > 5

        self.j = 1
        self.Power_change = 1.0
        self.Cb_avg = 0.0
        self.delta_time = delta_time
        self.delta_time_target = delta_time
        self.delta_time_temp = np.full(4, delta_time)

        self.ma_0 = 0.0
        self.mb_0 = nominal_mass_flow
        self.mc_0 = nominal_mass_flow

        self.time_calculating_transient_fuel_nodes = 0.0

        self.history = {
            "t": [0.0],
            "ma": [self.ma_0],
            "mb": [self.mb_0],
            "mc": [self.mc_0],
            "i_a": [],
            "i_b": [],
            "i_c": [],
            "Cb_a": [],
            "Cb_b": [],
            "Cb_c": [],
            "Ja": [],
            "Jb": [],
            "Jc": [],
            "T_b": [],
            "P_b": [Q_core_SS],
            "P_c": [-Q_core_SS],
            "Q_core": [Q_core_SS],
            "Cb_avg": [0.0],
            "T_avg_core": [],
            "T_avg_fuel": [],
        }
        if self.evaluate_CHF:
            self.history["MDNBR"] = []
        if evaluate_hot:
            self.history["i_b_hot"] = []
            self.history["T_b_hot"] = []

    def _f_A_minimize(self, H_core_tilde, f_A_val):
        return np.pi * H_core / (2 * H_core_tilde * np.sin(np.pi / 2 * H_core / H_core_tilde)) - f_A_val

    def _f_R_minimize(self, R_core_tilde, f_R_val):
        return 2.405 * (Dh_b / 2) / (2 * R_core_tilde * j1(2.405 * (Dh_b / 2) / R_core_tilde)) - f_R_val

    def initialize_steady_state(self, burnup_index=BURNUP_INDEX):
        self.burnup_index = burnup_index

        alpha_doppler_through_burnup = np.load(self.data_dir / "OPIUSV4_Doppler_coeffs.npy")
        self.alpha_doppler = alpha_doppler_through_burnup[self.burnup_index]

        MB_poly_coeffs = np.load(self.data_dir / "OPIUSV4_poly_coeffs.npy")
        self.MB_3rd_degree_polynomial_interpolation_coeffs = MB_poly_coeffs[self.burnup_index]

        if not burnup_dependent_delayed_neutrons:
            self.Lambda = 2.5e-5
            self.lambda_i = np.array([0.0124, 0.0305, 0.111, 0.301, 1.14, 3.01])
            self.beta_i = np.array([0.000215, 0.001424, 0.001274, 0.002568, 0.000748, 0.000273])
        else:
            self.Lambda = float(np.load(self.data_dir / "OPIUSV4_gen_times.npy")[self.burnup_index])
            self.lambda_i = np.load(self.data_dir / "OPIUSV4_lambda_is.npy")[self.burnup_index]
            self.beta_i = np.load(self.data_dir / "OPIUSV4_beta_is.npy")[self.burnup_index]
        self.beta = float(np.sum(self.beta_i))

        # -- Geometry and Channel Construction -- #
        R_core = Dh_b / 2.0
        self.channel_a, self.channel_b, self.channel_c, idx_core, lock_areas, n_interface = build_channels(self.HB_conc)
        self.n_interface_Ldl, self.n_interface_Udl = n_interface
        self.idx_start_core, self.idx_exit_core = idx_core
        self.A_LDl, self.A_UDl = lock_areas

        self.x_a, self.x_b, self.x_c = self.channel_a.x, self.channel_b.x, self.channel_c.x
        self.dx_a, self.dx_b, self.dx_c = self.channel_a.dx, self.channel_b.dx, self.channel_c.dx
        self.x_core = self.x_b[self.idx_start_core:self.idx_exit_core]
        self.A_a_array, self.A_b_array, self.A_c_array = self.channel_a.A, self.channel_b.A, self.channel_c.A

        # Peaking factors & extrapolated boundaries
        f_R_channel = np.load(self.data_dir / "OPIUSV4_max_radial_pfs.npy")[self.burnup_index]
        self.f_A = np.load(self.data_dir / "OPIUSV4_avg_axial_pfs.npy")[self.burnup_index]
        self.f_A_hot = np.load(self.data_dir / "OPIUSV4_pow_axial_pfs.npy")[self.burnup_index]
        self.f_R = f_R_channel * f_channel

        self.H_core_tilde = sp.optimize.fsolve(self._f_A_minimize, 1.2 * H_core, self.f_A)[0]
        self.H_core_tilde_hot = sp.optimize.fsolve(self._f_A_minimize, 1.2 * H_core, self.f_A_hot)[0]
        self.R_core_tilde = sp.optimize.fsolve(self._f_R_minimize, 1.2 * R_core, self.f_R)[0]

        # Fuel and lattice 
        self.number_pins = n_fuel_assembly * n_rods_assembly
        r_co = d_r / 2.0
        r_go = r_co - clad_thickness
        r_fo = r_go - gas_gap_thickness
        self.D_h = d_r * (4.0 / np.pi * (pitch / d_r)**2 - 1.0)
        self.Dh_c = np.sqrt(4.0 * self.A_c_array / np.pi)
        self.P_H = self.number_pins * 2.0 * np.pi * r_co
        self.V_fuel = self.number_pins * np.pi * (r_fo**2) * H_core

        average_fuel_volumetric_heating_SS_W = (Q_core_SS / self.V_fuel) * 1000.0
        self.peak_fuel_volumetric_heating_SS_W = average_fuel_volumetric_heating_SS_W * self.f_A * self.f_R
        self.peak_fuel_volumetric_heating_SS_W_hot = average_fuel_volumetric_heating_SS_W * self.f_A_hot * self.f_R

        # -- Source Terms -- #
        self.S_b_correction_factor = 1.0
        self.S_c_correction_factor = 1.0

        S_c_func_init = partial(S_c_func, x_c=self.x_c, S_c_correction_factor=self.S_c_correction_factor)
        S_b_func_init = partial(S_b_func, P_H=self.P_H)

        # initialize
        core_temp = np.array([
            Fuel(1, self.peak_fuel_volumetric_heating_SS_W / self.f_R *
                 np.cos(np.pi * (self.x_b[i] - (x_start_core + H_core / 2.0)) / self.H_core_tilde), None, None)
            if self.idx_start_core <= i < self.idx_exit_core else None
            for i in range(len(self.x_b))
        ], dtype=object)

        Total_P_b_SS = np.sum(S_b_func_init(core_temp, 0)[:-1]) * self.dx_b * A_core
        Total_P_c_SS = np.sum(S_c_func_init(0.0)[:-1]) * self.dx_c * A_c_SG
        self.S_b_correction_factor = Q_core_SS / Total_P_b_SS
        self.S_c_correction_factor = -Q_core_SS / Total_P_c_SS
        self.S_c_callable = partial(S_c_func, x_c=self.x_c, S_c_correction_factor=self.S_c_correction_factor)
        self.S_b_callable = partial(S_b_func, P_H=self.P_H)

        # post-correction, recreate the nodes from volumetric heating with the correction factor for proper initialization sequence
        max_steps = 10000
        self.core = np.array([
            Fuel(max_steps, self.peak_fuel_volumetric_heating_SS_W * self.S_b_correction_factor / self.f_R *
                 np.cos(np.pi * (self.x_b[i] - (x_start_core + H_core / 2.0)) / self.H_core_tilde), None, None)
            if self.idx_start_core <= i < self.idx_exit_core else None
            for i in range(len(self.x_b))
        ], dtype=object)

        if evaluate_hot:
            self.core_hot = np.array([
                Fuel(max_steps, self.peak_fuel_volumetric_heating_SS_W_hot * self.S_b_correction_factor *
                     np.cos(np.pi * (self.x_b[i] - (x_start_core + H_core / 2.0)) / self.H_core_tilde_hot), None, None)
                if self.idx_start_core <= i < self.idx_exit_core else None
                for i in range(len(self.x_b))
            ], dtype=object)

        # --Find steady state conditions at the initial parameters--#
        S_b_0 = self.S_b_callable(self.core, 0)
        S_c_0 = self.S_c_callable(0.0)

        self.i_a_initial = np.full(nx_a, self.HB_conc.h_T(T_pool))
        self.i_b_initial = np.empty(nx_b)
        self.i_b_initial[0] = self.HB_conc.h_T(T_l)
        self.i_a_initial[-self.n_interface_Ldl:] = self.i_b_initial[0]

        for i in range(1, nx_b):
            self.i_b_initial[i] = self.i_b_initial[i - 1] + S_b_0[i] * self.dx_b * self.A_b_array[i] / nominal_mass_flow
        self.i_a_initial[0:self.n_interface_Udl] = self.i_b_initial[-1]

        self.i_c_initial = np.empty(nx_c)
        self.i_c_initial[0] = self.i_b_initial[-1]
        for i in range(1, nx_c):
            self.i_c_initial[i] = self.i_c_initial[i - 1] + S_c_0[i] * self.dx_c * self.A_c_array[i] / nominal_mass_flow

        self.Cb_a_initial = np.full(nx_a, float(Boron_concentration_pool))
        self.Cb_a_initial[-self.n_interface_Udl:] = 0.0
        self.Cb_a_initial[:self.n_interface_Ldl] = 0.0
        self.Cb_b_initial = np.zeros(nx_b)
        self.Cb_c_initial = np.zeros(nx_c)

        self.channel_a.set_initial_profile(self.i_a_initial, self.Cb_a_initial)
        self.channel_b.set_initial_profile(self.i_b_initial, self.Cb_b_initial)
        self.channel_c.set_initial_profile(self.i_c_initial, self.Cb_c_initial)

        # Pressure loss calibrations
        rho_a_initial = self.channel_a.rho
        rho_b_initial = self.channel_b.rho
        rho_c_initial = self.channel_c.rho

        self.J_a_initial = self.channel_a.get_Jq()
        self.J_b_initial = self.channel_b.get_Jq()
        self.J_c_initial = self.channel_c.get_Jq()

        P_fl_b_initial = self.channel_b.get_friction_loss(nominal_mass_flow)
        P_acc_b_initial = self.channel_b.get_acc_loss()
        F_b_initial_th = g * (self.J_a_initial + self.J_b_initial) / (nominal_mass_flow**2)
        rho_core_initial = np.average(rho_b_initial[self.idx_start_core:self.idx_exit_core])
        P_spacer = F_b_initial_th - (P_fl_b_initial + P_acc_b_initial)
        self.zeta_spacer = 2.0 * rho_core_initial * P_spacer
        self.F_b = P_fl_b_initial + P_acc_b_initial + P_spacer

        P_fl_c = self.channel_c.get_friction_loss(nominal_mass_flow)
        P_acc_c = self.channel_c.get_acc_loss()
        self.zeta_steam_generator = self.zeta_spacer * 2.0
        Pl_SG = self.zeta_steam_generator / (np.average(rho_c_initial) * 2.0)
        self.F_c = P_fl_c + P_acc_c + Pl_SG

        zeta_exp = (1.0 - self.A_UDl / A_a_in)**2
        A_center = A_b * (0.62 + 0.38 * (self.A_LDl / A_a_in)**3)
        zeta_cont = (self.A_LDl / A_center - 1.0)**2
        P_Udl = zeta_exp / (2.0 * rho_a_initial[self.n_interface_Udl + 1] * self.A_UDl**2)
        P_Ldl = zeta_cont / (2.0 * rho_b_initial[-self.n_interface_Ldl - 1] * self.A_LDl**2)
        self.F_a = P_Udl + P_Ldl
        self.zeta_exp, self.zeta_cont = zeta_exp, zeta_cont

        # Hot Channel discrtization
        if evaluate_hot:
            S_b_hot_0 = self.S_b_callable(self.core_hot, 0)
            self.i_b_initial_hot = np.empty(nx_b)
            self.i_b_initial_hot[0] = self.i_b_initial[0]
            for i in range(1, nx_b):
                self.i_b_initial_hot[i] = self.i_b_initial_hot[i - 1] + S_b_hot_0[i] * self.dx_b * self.A_b_array[i] / nominal_mass_flow
            self.T_b_initial_hot = self.HB_conc.T_h(self.i_b_initial_hot)

            roughness_b = np.full(nx_b, roughness_else)
            roughness_b[self.idx_start_core:self.idx_exit_core] = roughness_core
            Dh_b_array = np.full(nx_b, Dh_b)
            Dh_b_array[self.idx_start_core:self.idx_exit_core] = self.D_h
            self.hot_channel = Channel1D("Hot Channel", L_b, nx_b, self.A_b_array, self.channel_b.theta, roughness_b, self.HB_conc, Dh_b_array)
            self.hot_channel.set_initial_profile(self.i_b_initial_hot, self.Cb_b_initial)

        # Steady-state fuel pin solve
        for i in range(len(self.core)):
            if self.core[i] is not None:
                self.core[i].T_lb[0] = self.channel_b.T[i]
                self.core[i].h[0] = HB40_h_correlations(self.core[i].T_lb[0], nominal_mass_flow, self.HB_conc, A_core, self.D_h, irradiated=self.irradiated)
                self.core[i].solve(-1)
                self.core[i].z = self.x_b[i]
                if evaluate_hot:
                    self.core_hot[i].T_lb[0] = self.T_b_initial_hot[i]
                    self.core_hot[i].h[0] = HB40_h_correlations(self.core_hot[i].T_lb[0], nominal_mass_flow, self.HB_conc, A_core, self.D_h, irradiated=self.irradiated)
                    self.core_hot[i].solve(-1)
                    self.core_hot[i].z = self.x_b[i]

        # Initial MDNBR evaluation
        if self.evaluate_CHF:
            heat_flux_hot_kWm2 = np.array([self.core_hot[self.idx_start_core + k].hf_co[0] / 1000.0 for k in range(self.idx_exit_core - self.idx_start_core)])
            CHF_initial = CHF_HB40(self.T_b_initial_hot[self.idx_start_core:self.idx_exit_core], P_initial, nominal_mass_flow, self.HB_conc, A_core, irradiated=self.irradiated)
            self.MDNBR_initial = np.min(CHF_initial / heat_flux_hot_kWm2)
            if self.MDNBR_initial < 1.0:
                print(f"CHF exceeded: MDNBR = {self.MDNBR_initial}")

        # point kinetics
        self.n0 = 10.0
        self.C0 = (self.beta_i * self.n0) / (self.lambda_i * self.Lambda)
        self.y0_point_kinetics = np.concatenate(([self.n0], self.C0))
        enthalpy_core = self.i_b_initial[self.idx_start_core:self.idx_exit_core]
        enthalpy_avg_core = np.sum(enthalpy_core[:-1]) * (self.x_core[1] - self.x_core[0]) / (self.x_core[-1] - self.x_core[0])
        self.T_avg_core_initial = self.HB_conc.T_h(enthalpy_avg_core)
        self.MB_normalization_constant = -p(self.T_avg_core_initial, SS_boron_concentration, self.MB_3rd_degree_polynomial_interpolation_coeffs)
        core_nodes = self.core[self.idx_start_core:self.idx_exit_core]
        T_fuel = np.array([node.average_fuel_T[0] for node in core_nodes])
        self.T_avg_fuel_initial = np.sum(T_fuel[:-1]) * (self.x_core[1] - self.x_core[0]) / (self.x_core[-1] - self.x_core[0])

        delta_Pb_0 = self.F_b * np.abs(self.mb_0) * self.mb_0
        delta_Pc_0 = self.F_c * np.abs(self.mc_0) * self.mc_0
        self.idx_exit_SG = np.abs(self.x_c - (x_start_SG + H_SG)).argmin()
        self.n_pump = int(self.idx_exit_SG + 1)
        self.h_pump_SS = 1.0 / (self.channel_c.rho[self.n_pump] * g) * ((delta_Pc_0 + delta_Pb_0) - g * (self.J_b_initial + self.J_c_initial))

        # transient variables
        self.t = [0.0]
        self.i_a = self.channel_a.i.copy()
        self.i_b = self.channel_b.i.copy()
        self.i_c = self.channel_c.i.copy()
        self.Cb_a = self.channel_a.Cb.copy()
        self.Cb_b = self.channel_b.Cb.copy()
        self.Cb_c = self.channel_c.Cb.copy()
        self.i_b_hot = self.i_b_initial_hot.copy() if evaluate_hot else None
        self.T_b_hot = self.T_b_initial_hot.copy() if evaluate_hot else None

        # Log t = 0
        self.history["i_a"].append(self.channel_a.i.copy())
        self.history["i_b"].append(self.channel_b.i.copy())
        self.history["i_c"].append(self.channel_c.i.copy())
        self.history["Cb_a"].append(self.channel_a.Cb.copy())
        self.history["Cb_b"].append(self.channel_b.Cb.copy())
        self.history["Cb_c"].append(self.channel_c.Cb.copy())
        self.history["Ja"].append(self.J_a_initial)
        self.history["Jb"].append(self.J_b_initial)
        self.history["Jc"].append(self.J_c_initial)
        self.history["T_b"].append(self.channel_b.T.copy())
        self.history["T_avg_core"].append(self.T_avg_core_initial)
        self.history["T_avg_fuel"].append(self.T_avg_fuel_initial)
        if self.evaluate_CHF:
            self.history["MDNBR"].append(self.MDNBR_initial)
        if evaluate_hot:
            self.history["i_b_hot"].append(self.i_b_initial_hot.copy())
            self.history["T_b_hot"].append(self.T_b_initial_hot.copy())

    def step(self):
        import time as pytime

        j = self.j

        # 1) Check the CFL limit and adjust delta t accordingly
        CFL_a = np.abs(self.ma_0) / (self.channel_a.rho * self.A_a_array) * self.delta_time / self.dx_a
        CFL_b = np.abs(self.mb_0) / (self.channel_b.rho * self.A_b_array) * self.delta_time / self.dx_b
        CFL_c = np.abs(self.mc_0) / (self.channel_c.rho * self.A_c_array) * self.delta_time / self.dx_c

        if np.any(CFL_a > 1):
            self.delta_time_temp[1] = self.dx_a * self.A_LDl * np.min(self.channel_a.rho) / np.abs(self.ma_0) * 0.8
        if np.any(CFL_b > 1):
            self.delta_time_temp[2] = self.dx_b * A_core * np.min(self.channel_b.rho) / np.abs(self.mb_0) * 0.8
        if np.any(CFL_c > 1):
            self.delta_time_temp[3] = self.dx_c * A_c_cl * np.min(self.channel_c.rho) / np.abs(self.mc_0) * 0.8
        self.delta_time = np.min(self.delta_time_temp)
        self.t.append(self.t[j - 1] + self.delta_time)

        # 2) solve the differential for the hydraulic grid
        t_eval = np.linspace(self.t[j - 1], self.t[j], 2)
        t_span = (self.t[j - 1], self.t[j])
        y0 = [self.ma_0, self.mb_0]
        hp_current = pump_head(self.t[j], self.h_pump_SS, pump_trip_transient=pump_trip_transient, feedwater_loss_transient=feedwater_loss_transient)
        sol = solve_ivp(
            derivatives_mq,
            t_span,
            y0,
            t_eval=t_eval,
            method="BDF",
            args=(hp_current, self.channel_a, self.channel_b, self.channel_c, self.F_a, self.F_b, self.F_c, self.channel_c.rho[self.n_pump]),
        )
        self.ma_0 = sol.y[0][-1]
        self.mb_0 = sol.y[1][-1]
        self.mc_0 = self.mb_0 - self.ma_0

        # 3) Density Locks 
        self.i_a, self.i_b, self.i_c, self.i_b_hot, self.Cb_a, self.Cb_b, self.Cb_c = solve_density_locks(
            self.ma_0, self.mb_0, self.mc_0,
            self.Cb_a, self.Cb_b, self.Cb_c,
            self.i_a, self.i_b, self.i_c,
            self.i_b_hot, evaluate_hot,
        )

        # 4) Enthapy and boron advection
        S_a = S_a_func(self.t[j])
        S_b = self.S_b_callable(self.core, j - 1)
        S_c = self.S_c_callable(self.t[j])

        self.i_a, self.Cb_a = self.channel_a.upwind_scheme(self.ma_0, self.delta_time, S_a, np.zeros(nx_a))
        self.i_b, self.Cb_b = self.channel_b.upwind_scheme(self.mb_0, self.delta_time, S_b, np.zeros(nx_b))
        self.i_c, self.Cb_c = self.channel_c.upwind_scheme(self.mc_0, self.delta_time, S_c, np.zeros(nx_c))
        self.T_b = self.HB_conc.T_h(self.i_b)

        if evaluate_hot:
            S_b_hot = self.S_b_callable(self.core_hot, j - 1)
            self.i_b_hot, _ = self.hot_channel.upwind_scheme(self.mb_0, self.delta_time, S_b_hot, np.zeros(nx_b))
            self.T_b_hot = self.HB_conc.T_h(self.i_b_hot)

        # 5)  Fuel Conduction
        t1 = pytime.time()
        Q_core = Q_core_SS * self.Power_change
        peak_fuel_volumetric_heating_W = (Q_core / self.V_fuel) * 1000.0 * self.f_A * self.f_R

        for i in range(len(self.core)):
            if self.core[i] is not None:
                self.core[i].T_lb[j] = self.T_b[i]
                self.core[i].h[j] = HB40_h_correlations(self.core[i].T_lb[j], self.mb_0, self.HB_conc, A_core, self.D_h, irradiated=self.irradiated)
                self.core[i].volumetric_heating[j] = (
                    peak_fuel_volumetric_heating_W / self.f_R
                    * np.cos(np.pi * (self.x_b[i] - (x_start_core + H_core / 2.0)) / self.H_core_tilde)
                    * self.S_b_correction_factor
                )
                self.core[i].t[j] = self.t[j]
                self.core[i].solve(j - 1)

                if evaluate_hot:
                    self.core_hot[i].T_lb[j] = self.T_b_hot[i]
                    self.core_hot[i].h[j] = HB40_h_correlations(self.core_hot[i].T_lb[j], self.mb_0, self.HB_conc, A_core, self.D_h, irradiated=self.irradiated)
                    self.core_hot[i].volumetric_heating[j] = (
                        peak_fuel_volumetric_heating_W
                        * np.cos(np.pi * (self.x_b[i] - (x_start_core + H_core / 2.0)) / self.H_core_tilde_hot)
                        * self.S_b_correction_factor
                    )
                    self.core_hot[i].t[j] = self.t[j]
                    self.core_hot[i].solve(j - 1)
        self.time_calculating_transient_fuel_nodes += (pytime.time() - t1)

        # 6) Pressure Losses
        P_spacer = self.zeta_spacer / (2.0 * np.average(self.channel_b.rho[self.idx_start_core:self.idx_exit_core]))
        self.F_b = self.channel_b.get_friction_loss(self.mb_0) + self.channel_b.get_acc_loss() + P_spacer
        Pl_SG = self.zeta_steam_generator / (np.average(self.channel_c.rho) * 2.0)
        self.F_c = self.channel_c.get_friction_loss(self.mc_0) + self.channel_c.get_acc_loss() + Pl_SG
        P_Udl = self.zeta_exp / (2.0 * self.channel_a.rho[self.n_interface_Udl + 1] * self.A_UDl**2)
        P_Ldl = self.zeta_cont / (2.0 * self.channel_b.rho[-self.n_interface_Ldl - 1] * self.A_LDl**2)
        self.F_a = self.channel_a.get_friction_loss(self.ma_0) + self.channel_a.get_acc_loss() + (P_Udl + P_Ldl)

        # 7) Point Kinetics
        Cb_core = self.Cb_b[self.idx_start_core:self.idx_exit_core]
        self.Cb_avg = np.sum(Cb_core[:-1]) * (self.x_core[1] - self.x_core[0]) / (self.x_core[-1] - self.x_core[0])
        Delta_Cb = self.Cb_avg - 0.0

        enthalpy_avg_core = np.sum(self.i_b[self.idx_start_core:self.idx_exit_core][:-1]) * (self.x_core[1] - self.x_core[0]) / (self.x_core[-1] - self.x_core[0])
        T_avg_core = self.HB_conc.T_h(enthalpy_avg_core)
        Delta_temp_core = T_avg_core - self.T_avg_core_initial

        core_nodes = self.core[self.idx_start_core:self.idx_exit_core]
        T_fuel = np.array([core_node.average_fuel_T[j - 1] for core_node in core_nodes])
        T_avg_fuel = np.sum(T_fuel[:-1]) * (self.x_core[1] - self.x_core[0]) / (self.x_core[-1] - self.x_core[0])
        Delta_temp_fuel = T_avg_fuel - self.T_avg_fuel_initial

        if nonlinear_PK:
            sol_point_kinetics = solve_ivp(
                point_kinetics_nonlinear,
                t_span,
                self.y0_point_kinetics,
                t_eval=t_eval,
                args=(
                    self.Cb_avg + SS_boron_concentration,
                    T_avg_core,
                    self.MB_3rd_degree_polynomial_interpolation_coeffs,
                    self.MB_normalization_constant,
                    Delta_temp_fuel,
                    self.alpha_doppler,
                    self.beta,
                    self.Lambda,
                    self.beta_i,
                    self.lambda_i,
                ),
                method="Radau",
            )
        else:
            sol_point_kinetics = solve_ivp(
                point_kinetics_linear,
                t_span,
                self.y0_point_kinetics,
                t_eval=t_eval,
                args=(
                    Delta_Cb,
                    alpha_B,
                    Delta_temp_core,
                    alpha_M,
                    Delta_temp_fuel,
                    self.alpha_doppler,
                    self.beta,
                    self.Lambda,
                    self.beta_i,
                    self.lambda_i,
                ),
                method="Radau",
            )
        n_final = sol_point_kinetics.y[0, -1]
        self.Power_change = max(n_final / self.n0, 1.9e-2)
        self.C0 = sol_point_kinetics.y[1:, -1]
        self.y0_point_kinetics = np.concatenate(([n_final], self.C0))

        # 8) Thermal Margins
        if self.evaluate_CHF:
            heat_flux_hot_t = np.array([self.core_hot[self.idx_start_core + k].hf_co[j] / 1000.0 for k in range(self.idx_exit_core - self.idx_start_core)])
            CHF_t = CHF_HB40(self.T_b_hot[self.idx_start_core:self.idx_exit_core], P_initial, self.mb_0, self.HB_conc, A_core, irradiated=self.irradiated)
            MDNBR_t = np.min(CHF_t / heat_flux_hot_t)
            if MDNBR_t < 1.0:
                print(f"CHF exceeded at t={self.t[j]:.2f} s: MDNBR = {MDNBR_t:.3f}")
            self.history["MDNBR"].append(MDNBR_t)

        # 9) Reset delta_time to the target value and save results
        self.delta_time = self.delta_time_target
        self.history["t"].append(self.t[j])
        self.history["ma"].append(self.ma_0)
        self.history["mb"].append(self.mb_0)
        self.history["mc"].append(self.mc_0)
        self.history["i_a"].append(self.i_a.copy())
        self.history["i_b"].append(self.i_b.copy())
        self.history["i_c"].append(self.i_c.copy())
        self.history["Cb_a"].append(self.Cb_a.copy())
        self.history["Cb_b"].append(self.Cb_b.copy())
        self.history["Cb_c"].append(self.Cb_c.copy())
        self.history["Ja"].append(self.channel_a.get_Jq())
        self.history["Jb"].append(self.channel_b.get_Jq())
        self.history["Jc"].append(self.channel_c.get_Jq())
        self.history["Cb_avg"].append(self.Cb_avg)
        self.history["T_b"].append(self.T_b.copy())
        self.history["P_b"].append(np.sum(self.S_b_callable(self.core, j)[:-1]) * self.dx_b * A_core)
        self.history["P_c"].append(np.sum(S_c[:-1]) * self.dx_c * A_c_SG)
        self.history["Q_core"].append(Q_core)
        self.history["T_avg_core"].append(T_avg_core)
        self.history["T_avg_fuel"].append(T_avg_fuel)
        if evaluate_hot:
            self.history["i_b_hot"].append(self.i_b_hot.copy())
            self.history["T_b_hot"].append(self.T_b_hot.copy())

        self.j += 1

    def finalize(self):
        # Converts lists to numpy arrays and crops fuel memory
        for key, values in self.history.items():
            if len(values) > 0:
                self.history[key] = np.asarray(values)

        for i in range(len(self.core)):
            if self.core[i] is not None:
                self.core[i].crop(self.j)
                if evaluate_hot:
                    self.core_hot[i].crop(self.j)