import numpy as np

# -- Simulation control --

Time = 60 # s, target time
delta_time = 0.25 # s, hopeful time interval, must be a float (1.0 instead of 1)
evaluate_hot = True
evaluate_CHF = True
pump_trip_transient = True
feedwater_loss_transient = False
nonlinear_PK = True # True to use p(T_M, C_B) and False to use alpha_B and alpha_M
burnup_dependent_delayed_neutrons = True # if False => uses delayed neutron parameters for U-235 https://www.researchgate.net/figure/Parameters-of-Six-Delayed-Neutron-Groups_tbl1_282362969
BURNUP_INDEX = 5 # 0 (BOC) to 10 (EOC)
enable_plots = True

# -- Channel geometry and discretization --

g = 9.81 #m/s^2
L_a = 27.652 # [m]  Channel a length
L_b = 22.948 # [m] Channel b length
L_c = 123.448 # [m] Channel c length
# Spatial Mesh
nx_a = 260
nx_b = 100
nx_c = 100
# sections
A_a_in = 80.0 #m^2, Flow area pool
roughness_else = 0.07 * 1e-3 #m
Dh_b = 3.75 #m Equivalent core diameter 
A_core = 6.065 #m^2 (checked) flow area through the core region
A_b = (Dh_b / 1.15)**2 / 4 *np.pi # m^2 flow area riser , assumed from drawing
roughness_core = 0.004 * 1e-3 #m

# -- Geometry of components --

# density locks 
n_tubes_LDl = 2000 # number of tubes per bundle in lower density lock
n_tubes_UDl = 4000 # number of tubes per bundle in lower density lock
id_tube = 55 #[mm] inner diameter tube in density lock 
L_Udl = 1 # m, Upper density lock length 
L_Ldl = 1# m LDL length
L_lower_plenum = 2.352 #m
# channel C
A_c_hl = 0.665 # [m^2]
A_c_SG = 4.406 # [m^2]
A_c_cl = 0.635 # [m^2]
A_c_downcomer = 2.5 # [m^2]
# core and SG
H_core = 2.5 # [m] Active core height 
H_SG = 15 # [m] Active SG length
x_start_core = 1  # Core placement from bottom of channel B [m]
x_start_SG = 38.25  # SG placement from top of channel C [m]


# -- Plant configuration --

Q_core_SS = 2e6 # [kW], also equal to HX heat removal power in SS 
nominal_mass_flow = 15000 # [Kg/s], nominal s.s. mass flow rate in the primary loop 
P_initial = 10 # bar, system pressure in s.s.
T_l = 324.0 # °C # T lower lock in channel b 
T_u = 290.0 # °C T at upper lock in channel b 
T_pool = 50.0 # °C 
SS_boron_concentration = 0 # ppm
Boron_concentration_pool = 2200 - SS_boron_concentration  # ppm, the whole boron concentration is shifted by SS_boron_concentration so numerical SS core boron concentration is 0
alpha_B = -13.1 # [pcm / ppm] boron coefficient
alpha_M = -13.5  # [pcm/ C°] moderator temperature coefficient
HB_wt = 30 # weight percentage of High boilers

# -- Fuel assembly and pin layout --

n_fuel_assembly = 213 # [-] 
n_rods_assembly = 316 #[-] 
d_r = 0.0095 # [m] , rod outer diameter
clad_thickness = 0.000572 # [m]
gas_gap_thickness = 0.000083 # [m]
pitch = 0.0126 # [m], lattice pitch 
f_channel = 1.2 # radial, within channel, peaking factor
