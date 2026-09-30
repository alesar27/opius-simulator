import numpy as np

def solve_density_locks(m_a, m_b, m_c, Cb_a, Cb_b, Cb_c, i_a, i_b, i_c, i_b_hot=None, evaluate_hot=False):
    update_hot = evaluate_hot and (i_b_hot is not None)
    if m_a == 0 :
            i_b [0] = i_c [-1] 
            if update_hot : i_b_hot[0] = i_b[0]
            i_c [0] = i_b [-1]  # upper lock 2 unknowns in 1 equation, if temperature is the same mass and energy conservation is ok
            i_a [0] = i_a [0] 
            #boron
            Cb_a [0] = Cb_a [0]
            Cb_b [0] = 0
            Cb_c [0] = 0
    elif (m_a > 0, m_b > 0, m_c > 0) == (True, True, True):
        i_b [0] = (i_a[-1] * m_a + i_c [-1] * m_c) / np.abs(m_b) # energy conservation at lower lock
        if update_hot : i_b_hot[0] = i_b[0]
        i_c [0] = i_b [-1] 
        i_a [0] = i_b [-1] 
        #boron
        Cb_b [0] = (Cb_a [-1] * m_a + Cb_c [-1] * m_c) / np.abs(m_b) #mass conservation at lower lock
        Cb_c [0] = Cb_b [-1] 
        Cb_a [0] = Cb_b [-1]
    elif (m_a < 0, m_b > 0, m_c > 0) == (True, True, True):
        i_c [0] = (i_b[-1] * m_b + i_a [0] * abs(m_a)) / np.abs(m_c)# energy conservation at upper lock
        i_b [0] = i_c [-1] 
        if update_hot : i_b_hot[0] = i_b[0]
        i_a [-1] = i_c [-1] 
        #boron
        Cb_c [0] = (Cb_b[-1] * m_b + Cb_a [0] * abs(m_a)) / np.abs(m_c) #mass conservation at upper lock
        Cb_b [0] = Cb_c [-1] 
        Cb_a [-1] = Cb_c [-1]
    elif (m_a > 0, m_b > 0, m_c < 0) == (True, True, True):
        i_a [0] = (i_b[-1] * m_b  + i_c [0] * abs(m_c)) / np.abs(m_a )
        i_b [0] = i_a [-1]
        if update_hot : i_b_hot[0] = i_b[0]
        i_c [-1] = i_a [-1]  
        #boron
        Cb_a [0] = (Cb_b[-1] * m_b + Cb_c [0] * abs(m_c)) / np.abs(m_a)
        Cb_b [0] = Cb_a [-1]
        Cb_c[-1] = Cb_a [-1] 
    elif (m_a > 0, m_b < 0, m_c < 0) == (True, True, True):
        i_c [-1] = (i_b[0] * abs(m_b) + i_a [-1] * m_a) / np.abs(m_c)
        i_b [-1] = i_c [0]
        if update_hot : i_b_hot[-1] = i_b[-1]
        i_a [0] = i_c [0] 
        #boron
        Cb_c [-1] = (Cb_b[0] * abs(m_b) + Cb_a [-1] * m_a) / np.abs(m_c)
        Cb_b [-1] = Cb_c [0]
        Cb_a[0] = Cb_c [0]
    return i_a, i_b, i_c, i_b_hot, Cb_a, Cb_b, Cb_c