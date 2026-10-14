import numpy as np

class HB40():
    # T is in °C , HB is in weight %
    def __init__(self, HB):
        self.HB = HB

    def rho(self, T):
        # T in °C, returns kg/m^3
        return 1.0181e3 - 7.3891e-1 * T + 2.2343 * self.HB + 1.667e-3 * T * self.HB
    
    def my(self, T):
        # my is in N * s  * m^-2
        if T < 150 :
            T = T + 273.15 # T is in K
            return np.exp(1.2099 + 4.3339e-1 * self.HB - 1.0459e-2 * self.HB**2 - 8.3247e3 * T**(-1) - 3.2463e2 * self.HB * T**(-1) + 7.2201 * self.HB**2 * T**(-1) + 2.1478e6 * T**(-2) + 5.9661e4 * self.HB * T**(-2) - 1.0455e3 * self.HB**2 * T**(-2)) 
        else:
            T = T + 273.15 # T is in K
            return np.exp(-11.626 + 2.6269e-2 * self.HB - 5.3368e-4 * self.HB**2 + 2.1856e3 * T**(-1) - 1.156e1 * self.HB * T**(-1) + 5.4814e-1 * self.HB**2 * T**(-1))
        
    def k(self, T):
        # returns in W /(m * K)
        return 1.2677e-1 - 1.0864e-4 * T + 4.862e-4 * self.HB + 7.8045e-8 * T * self.HB
    
    def c_p(self, T):
        #returns in kJ / (kg * K) 
        return (1.5049e3 + 3.5476 * T + 5.0183e-1 * self.HB - 1.9283e-2 * T * self.HB) * 1e-3

    def h_T(self, T):
        # returns in kJ/kg
        a = (3.5476 / 2 - 1.9283e-2 / 2 * self.HB) * 1e-3
        b = (1.5049e3 + 5.0183e-1 * self.HB) * 1e-3
        return (a * T ** 2 + b * T)
    
    def T_h(self, h):
        # returns in kJ/kg
        a = (3.5476 / 2 - 1.9283e-2 / 2 * self.HB) * 1e-3
        b = (1.5049e3 + 5.0183e-1 * self.HB) * 1e-3
        return (-b + np.sqrt(b ** 2 + 4 * a * h)) / (2 * a)
    
    def c_p_int(self, T_cold, T_hot):
        return (1.5049e3 + 5.0183e-1 * self.HB) * (T_hot - T_cold) + (3.5476 - 1.9283e-2 * self.HB) * (T_hot**2 - T_cold**2) / 2