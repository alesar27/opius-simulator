import numpy as np
from numpy import log as ln
from numpy import exp
from matplotlib import pyplot as plt
import time
import pprint

# The code is translated to °C respective to previous fuel conduction ones.

class Fuel():
    def __init__(self, number_time_steps, volumetric_heating_SS, h_SS, T_lb_SS, initial_time=0):
        # geometric parameters
        self.r_co = 0.0095 / 2 # [m]
        self.r_go = 0.0095 / 2 - 0.000572 # [m]
        self.delta_r_g = 0.083e-3 #[m]
        self.r_fo = self.r_go - self.delta_r_g # [m]
        # radial saptial discretization
        self.number_fuel_nodes = 15
        self.number_gap_nodes = 5
        self.number_clad_nodes = 10
        self.number_nodes = self.number_fuel_nodes + self.number_gap_nodes + self.number_clad_nodes
        self.N = self.number_nodes - 1
        self.number_time_steps = number_time_steps
        self.k = np.zeros(self.number_nodes)
        self.T = np.zeros((self.number_time_steps, self.number_nodes))
        self.C = np.zeros(self.number_nodes)      
        self.volumetric_heating = np.zeros(self.number_time_steps)  # can be set to force a power evolution
        self.volumetric_heating[0] = volumetric_heating_SS
        self.h = np.zeros(self.number_time_steps)   
        self.h[0] = h_SS
        self.T_lb = np.zeros(self.number_time_steps)
        self.T_lb[0] = T_lb_SS # [°C]
        self.hf_co = np.zeros(self.number_time_steps)
        self.hf_co[0] = volumetric_heating_SS * self.r_fo**2 / (2 * self.r_co) 

        self.q = np.zeros((self.number_nodes, self.number_time_steps))
        self.T_centerline = np.zeros(self.number_time_steps) 
        self.average_fuel_T = np.zeros(self.number_time_steps)
        
        # mesh construction and initial/guess parameter setting
        center = 0.0001
        L_fuel = np.linspace(ln(center), ln(self.r_fo), self.number_fuel_nodes + 1)
        xi_fuel = np.zeros(self.number_fuel_nodes)
        for i in range(0, self.number_fuel_nodes):
            xi_fuel[i] = (L_fuel[i] + L_fuel[i+1]) / 2.0
            self.T[0][i] = 1200 # initialization value, can be optimized
            self.k[i] = self.k_fuel_func(self.T[0][i])
            self.C[i] = self.C_fuel_func(self.T[0][i])
            self.q[i] = self.volumetric_heating # array
            self.q[i][0] = volumetric_heating_SS  # initialization value

        L_gap = np.linspace(ln(self.r_fo), ln(self.r_go), self.number_gap_nodes + 1)
        xi_gap = np.zeros(self.number_gap_nodes)
        for i in range(0, self.number_gap_nodes):
            xi_gap[i] = (L_gap[i] + L_gap[i+1]) / 2.0
            self.T[0][self.number_fuel_nodes + i] = 800  # initialization value, can be optimized
            self.k[self.number_fuel_nodes + i] = self.k_gap_func(self.T[0][self.number_fuel_nodes + i])
            self.C[self.number_fuel_nodes + i] = 0 # neglected

        L_clad = np.linspace(ln(self.r_go), ln(self.r_co) + abs(ln(self.r_co) - ln(self.r_go)) / (2 * self.number_clad_nodes - 1), self.number_clad_nodes+1)
        xi_clad = np.zeros(self.number_clad_nodes)
        for i in range(0, self.number_clad_nodes):
            xi_clad[i] = (L_clad[i] + L_clad[i+1]) / 2.0
            self.T[0][self.number_fuel_nodes + self.number_gap_nodes + i] = 600 # initialization value, can be optimized
            self.k[self.number_fuel_nodes + self.number_gap_nodes + i] = self.k_clad_func(self.T[0][self.number_fuel_nodes + self.number_gap_nodes + i])
            self.C[self.number_fuel_nodes + self.number_gap_nodes + i] = 2.23e6
        
        self.L = np.concatenate([L_fuel, L_gap[1:], L_clad[1:]])
        self.xi = np.concatenate([xi_fuel, xi_gap, xi_clad])

        self.r = np.exp(self.xi) # for plotting purposes
        self.z = None # for plotting purposes

        self.t = np.zeros(number_time_steps) # to be filled beforehand or on the spot but need t[n+1] for computations
        self.t[0] = initial_time

        # calculation speed up
        self.exp2L = np.exp(2 * self.L)


    def solve(self, n): # n is the time index. -1 for S.S. and other for transient. Will solve value for time index n+1
        A = np.zeros((self.number_nodes, self.number_nodes))
        bvec = np.zeros(self.number_nodes) 
        # iterations for solution convergence (1 iteration is explicit)
        if n==-1:
            number_iterations = 10
        if n!=-1:
            number_iterations = 1

        if self.h[n+1] == 0:# or self.h[n] == 0:
            print(f"Heat transfer coefficient set to 0")
        if self.T_lb[n+1] == 0:# or self.T_lb[n] == 0:
            print(f"Liquid bulk temperature set to 0")
        if self.volumetric_heating[n+1] == 0:# or self.volumetric_heating[n] == 0:
            print(f"Volumetric heating set to 0")
        if self.t[n+1] == 0 and n!=-1:
            print(f"Time set to 0")

        # setting q[n+1] in the fuel to volumetric_heating[n+1]
        for i in range(0, self.number_fuel_nodes):
            self.q[i][n+1] = self.volumetric_heating[n+1]  # initialization value

        
        for iteration in range(number_iterations):
            # Applying boundary conditions in A and bvec
            # fuel centerline Neumann
            k_0plushalf = k_iplushalf_func(self.xi[0], self.xi[1], self.L[1], self.k[0], self.k[1])
            A[0][0] = -2.0 * k_0plushalf / (self.xi[1] - self.xi[0]) # b0
            A[0][1] = c(k_0plushalf, self.xi[0], self.xi[1]) # c0
            if n==-1:
                bvec[0] = -self.exp2L[1] * self.q[0][0] # d0
                # outer cladding temperature Dirichlet
                """A[-1][-1] = 1
                bvec[-1] = self.T_lb[0] + self.hf_co[0] / self.h[0]"""
                # SS Robin BC — enforces conduction = convection at outer surface
                k_Nminushalf = k_iminushalf_func(self.xi[self.N-1], self.xi[self.N], 
                                                self.L[self.N], self.k[self.N-1], self.k[self.N])
                A[-1][-1] = 2*exp(self.xi[self.N])*self.h[0] + 2*k_Nminushalf/(self.xi[self.N]-self.xi[self.N-1])
                A[-1][-2] = -2*k_Nminushalf/(self.xi[self.N]-self.xi[self.N-1])
                bvec[-1]  = 2*exp(self.xi[self.N])*self.h[0]*self.T_lb[0]

                # general system construction
                for i in range(1, self.N):
                    k_iplushalf = k_iplushalf_func(self.xi[i], self.xi[i+1], self.L[i+1], self.k[i], self.k[i+1])
                    k_iminushalf = k_iminushalf_func(self.xi[i-1], self.xi[i], self.L[i], self.k[i-1], self.k[i])
                    A[i][i-1] = a(k_iminushalf, self.xi[i-1], self.xi[i])
                    A[i][i] = bprime(k_iminushalf, k_iplushalf, self.xi[i-1], self.xi[i], self.xi[i+1])
                    A[i][i+1] = c(k_iplushalf, self.xi[i], self.xi[i+1])
                    bvec[i] = dprime(self.exp2L[i], self.exp2L[i+1], self.q[i][n+1])
            else:
                A[0][0] = -2.0 * k_0plushalf / (self.xi[1] - self.xi[0]) - self.exp2L[1] * self.C[0] / (self.t[n+1] - self.t[n])
                bvec[0] = -self.exp2L[1] * self.q[0][n+1] - self.exp2L[1] * self.C[0] * self.T[n][0] / (self.t[n+1] - self.t[n])

                # outer cladding Robin convection
                k_Nminushalf = k_iminushalf_func(self.xi[self.N-1], self.xi[self.N], self.L[self.N], self.k[self.N-1], self.k[self.N])
                A[-1][-1] = 2 * exp(self.xi[self.N]) * self.h[n+1] + 2 * k_Nminushalf / (self.xi[self.N] - self.xi[self.N-1]) + (exp(2 * self.xi[self.N]) - exp(2 * self.L[self.N])) * self.C[self.N] / (self.t[n+1] - self.t[n])
                A[-1][-2] = -2 * k_Nminushalf / (self.xi[self.N] - self.xi[self.N-1])
                bvec[-1] = 2 * exp(self.xi[self.N]) * self.h[n+1] * self.T_lb[n+1] + (exp(2 * self.xi[self.N]) - exp(2 * self.L[self.N])) * self.C[self.N] * self.T[n][self.N] / (self.t[n+1] - self.t[n])
                # general system construction
                for i in range(1, self.N):
                    k_iplushalf = k_iplushalf_func(self.xi[i], self.xi[i+1], self.L[i+1], self.k[i], self.k[i+1])
                    k_iminushalf = k_iminushalf_func(self.xi[i-1], self.xi[i], self.L[i], self.k[i-1], self.k[i])
                    A[i][i-1] = a(k_iminushalf, self.xi[i-1], self.xi[i])
                    A[i][i] = b(k_iminushalf, k_iplushalf, self.xi[i-1], self.xi[i], self.xi[i+1], self.exp2L[i], self.exp2L[i+1], self.C[i], self.t[n], self.t[n+1])
                    A[i][i+1] = c(k_iplushalf, self.xi[i], self.xi[i+1])
                    bvec[i] = d(self.exp2L[i], self.exp2L[i+1], self.q[i][n+1], self.C[i], self.T[n][i], self.t[n], self.t[n+1])
            
            self.T[n+1] = np.linalg.solve(A, bvec)

            # generate thermal conductivities and specific heats for next iteration or next step
            if iteration != number_iterations:
                self.k[0:self.number_fuel_nodes] = self.k_fuel_func(self.T[n+1][0:self.number_fuel_nodes])
                self.C[0:self.number_fuel_nodes] = self.C_fuel_func(self.T[n+1][0:self.number_fuel_nodes])

                self.k[self.number_fuel_nodes:self.number_fuel_nodes+self.number_gap_nodes] = self.k_gap_func(self.T[n+1][self.number_fuel_nodes:self.number_fuel_nodes+self.number_gap_nodes])

                self.k[self.number_fuel_nodes+self.number_gap_nodes:self.number_nodes] = self.k_clad_func(self.T[n+1][self.number_fuel_nodes+self.number_gap_nodes:self.number_nodes])

            # Delta_T_numerical = self.T[n+1][0] - self.T[n+1][-1]
            
            # outer heat flux
            self.hf_co[n+1] = self.h[n+1] * (self.T[n+1][self.N] - self.T_lb[n+1])

            # average fuel temperature calculation
            numerator = 0
            denominator = self.r_fo**2
            for i in range(self.number_fuel_nodes):
                numerator += (self.exp2L[i+1] - self.exp2L[i]) * self.T[n+1][i]
            self.average_fuel_T[n+1] = numerator / denominator

            # center line temperature (approximation)
            self.T_centerline[n+1] = self.T[n+1][0]

    def k_fuel_func(self, T):
        smallt = (T+273.15) / 1000
        return 100 / (7.5408 + 17.692 * smallt + 3.6142 * smallt**2) + (6400 / smallt**(5/2)) * exp(-16.35 / smallt)
    
    def C_fuel_func(self, T):
        smallt = (T+273.15) / 1000
        return 2013831.36092593 + 3394745.72777778 * smallt - 3251550.45796296 * smallt**2 + 1217462.78888889 * smallt**3 - 101644.36333333 * smallt**4 - 27555.60394444 * smallt**(-2)
    
    def k_gap_func(self, T):
        T_Kel = T + 273.15
        return 0.0635 + 0.00031 * T_Kel - 0.244 * 1e-7 * T_Kel**2
    
    def k_clad_func(self, T):
        T_Kel = T + 273.15
        return 11.3 + 225e-5 * T + 72.5e-8 * T**2
    
    def resize(self, extension):
        self.number_time_steps = self.number_time_steps + extension
        self.t = np.pad(self.t, (0, extension))
        self.volumetric_heating = np.pad(self.volumetric_heating, (0, extension))
        self.T = np.pad(self.T, ((0, extension), (0, 0)))
        self.h = np.pad(self.h, (0, extension))
        self.T_lb = np.pad(self.T_lb, (0, extension))
        self.hf_co = np.pad(self.hf_co, (0, extension))
        self.q = np.pad(self.q, (0, extension))
        self.T_centerline = np.pad(self.T_centerline, (0, extension))
        self.average_fuel_T = np.pad(self.average_fuel_T, (0, extension))

    def crop(self, j):
        self.number_time_steps = j
        self.t = self.t[:j]
        self.volumetric_heating = self.volumetric_heating[:j]
        self.T = self.T[:j]
        self.h = self.h[:j]
        self.T_lb = self.T_lb[:j]
        self.hf_co = self.hf_co[:j]
        self.q = self.q[:j]
        self.T_centerline = self.T_centerline[:j]
        self.average_fuel_T = self.average_fuel_T[:j]


def k_iminushalf_func(xi_iminus1, xi_i, L_i, k_iminus1, k_i):
    return (xi_i - xi_iminus1) / ((xi_i - L_i) / k_i + (L_i - xi_iminus1) / k_iminus1)

def k_iplushalf_func(xi_i, xi_iplus1, L_iplus1, k_i, k_iplus1):
    return (xi_iplus1 - xi_i) / ((xi_iplus1 - L_iplus1) / k_iplus1 + (L_iplus1 - xi_i) / k_i)

def a(k_iminushalf, xi_iminus1, xi_i):
    return 2 * k_iminushalf / (xi_i - xi_iminus1)

def bprime(k_iminushalf, k_iplushalf, xi_iminus1, xi_i, xi_iplus1):
    return - (2 * k_iminushalf / (xi_i - xi_iminus1) + 2 * k_iplushalf / (xi_iplus1 - xi_i))

def b(k_iminushalf, k_iplushalf, xi_iminus1, xi_i, xi_iplus1, exp2L_i, exp2L_iplus1, C_i, t_n, t_nplus1):
    return - (2 * k_iminushalf / (xi_i - xi_iminus1) + 2 * k_iplushalf / (xi_iplus1 - xi_i)) - (exp2L_iplus1 - exp2L_i) * C_i / (t_nplus1 - t_n)

def c(k_iplushalf, xi_i, xi_iplus1):
    return 2 * k_iplushalf / (xi_iplus1 - xi_i)

def dprime(exp2L_i, exp2L_iplus1, q_i):
    return (exp2L_i - exp2L_iplus1) * q_i

def d(exp2L_i, exp2L_iplus1, q_n, C_i, Tn_i, t_n, t_nplus1):
    return (exp2L_iplus1 - exp2L_i) * (-q_n - C_i * Tn_i / (t_nplus1 - t_n))
