import numpy as np
from numba import types, njit
from numba.typed import List

def get_param_sample(p_mean,p_range,N):
    return p_mean + p_range * np.random.uniform(-1, 1, N)

def get_const_param(lambda_i,omega_i):
    return (lambda_i - 1.0) + 1j * omega_i

def get_IC(Z0,Z0_std,N):
    return (np.random.normal(Z0.real, Z0_std, N) + 1j * np.random.normal(Z0.imag, Z0_std, N))

def get_coupling_matrix(beta,R,C,normalize_weight=False):
    """
    beta -> coupling strength
    R    -> structural coupling matrix (number of fibers)
    C    -> functional coupling matrix (e.g., rs-fMRI correlation matrix)
    normalize_weight -> if set, then beta -> beta(i)=beta/n(i), where n(i) is the number of inputs of node i
    """
    K = beta * (R * C) # element-wise R * C
    if normalize_weight:
        for i in range(K.shape[0]):
            n_inp  = len(np.nonzero(K[i,:])[0])
            K[i,:] = K[i,:] / n_inp
    return np.ascontiguousarray(K.astype(np.complex128))

@njit(types.complex128[:](types.complex128[:], types.complex128[:]))
def f_Hopf_vec(z, Complex_Param_i):
    """The local dynamics function f(z_i) for complex z."""
    abs_sq = np.abs(z)**2
    abs_quad = abs_sq**2
    return Complex_Param_i * z + 2.0 * z * abs_sq - z * abs_quad

@njit(
    types.complex128[:,:](
        types.float64, types.float64, types.float64, types.int64, types.float64,
        types.complex128[:], types.complex128[:], types.complex128[:,:] # K_matrix_ij type changed here
    )
)
def integrate_Hopf_oscillators_const_delay(
    T, dt, alpha, N, delay_steps_float,
    Z_initial, Complex_Param_i, K_matrix_ij
):
    # --- Pre-calculations inside the function ---
    Nt = types.int64(T / dt)
    delay_steps = types.int64(delay_steps_float)
    buffer_size = delay_steps + 1
    
    sqrt_dt_complex_noise = np.sqrt(dt / 2.0)
    
    # Storage and history buffer initialization (Numba-compatible)
    Z = np.zeros((Nt, N), dtype=np.complex128)
    Z[0, :] = Z_initial
    history_buffer = np.zeros((buffer_size, N), dtype=np.complex128)
    for i in range(buffer_size):
        history_buffer[i, :] = Z_initial
    history_index = 0

    # Pre-calculate the row sums of the coupling matrix K_ij
    # This is Sum_j K_ij, which is constant throughout the simulation
    # The sum of a complex matrix results in a complex vector (dtype=types.complex128)
    row_sums_K = np.sum(K_matrix_ij, axis=1)

    # --- Simulation Loop (Complex Euler-Maruyama Method) ---
    for t in range(Nt - 1):
        Z_delayed = history_buffer[history_index, :]
        Z_current = Z[t, :]

        # --- CORRECTED COUPLING TERM CALCULATION ---
        
        # 1. Delayed Input Term: Sum_j K_ij * z_j(t - tau)
        # K_matrix_ij (types.complex128) @ Z_delayed (types.complex128) -> COMPATIBLE
        delayed_input = K_matrix_ij @ Z_delayed
        
        # 2. Current Term: z_i(t) * Sum_j K_ij
        # Z_current (types.complex128) * row_sums_K (types.complex128) -> COMPATIBLE
        current_dependent_term = Z_current * row_sums_K
        
        # Total coupling term D_i = (Delayed Input) - (Current Term)
        coupling_term = delayed_input - current_dependent_term
        
        # --- END CORRECTED COUPLING TERM CALCULATION ---
        
        drift_term = f_Hopf_vec(Z_current, Complex_Param_i) + coupling_term

        # Noise Term
        dW_R = np.random.normal(0.0, 1.0, N) * sqrt_dt_complex_noise 
        dW_I = np.random.normal(0.0, 1.0, N) * sqrt_dt_complex_noise
        dW_complex = dW_R + 1j * dW_I
        noise_term = alpha * dW_complex
        
        # Euler-Maruyama Step
        Z[t + 1, :] = Z_current + drift_term * dt + noise_term

        # Update History Buffer
        history_index = (history_index + 1) % buffer_size
        history_buffer[history_index, :] = Z[t + 1, :]
        
    return Z


_type_neighbor_index_list        = types.int64[:]
_type_delay_list                 = types.int64[:]
_type_coupling_strength_list     = types.complex128[:]
_type_all_neighbors_index_list   = types.ListType(_type_neighbor_index_list)
_type_all_delays_list            = types.ListType(_type_delay_list)
_type_all_coupling_strength_list = types.ListType(_type_coupling_strength_list)
_type_get_coupling_lists_output  = types.Tuple((_type_all_neighbors_index_list,_type_all_coupling_strength_list,_type_all_delays_list))
@njit(_type_get_coupling_lists_output(types.complex128[:,:],types.int64[:,:]))
def get_coupling_lists(K,T):
    neigh_list = List.empty_list(_type_neighbor_index_list)
    coupl_list = List.empty_list(_type_coupling_strength_list)
    delay_list = List.empty_list(_type_delay_list)
    N = K.shape[0]
    for i in range(N):
        j_lst = np.nonzero(K[i,:])[0]
        K_val = np.zeros(j_lst.size,types.complex128)
        t_val = np.zeros(j_lst.size,types.int64)
        for m,j in enumerate(j_lst):
            K_val[m] = K[i,j]
            t_val[m] = T[i,j]
        neigh_list.append(j_lst)
        coupl_list.append(K_val)
        delay_list.append(t_val)
    return neigh_list,coupl_list,delay_list

@njit(types.complex128(types.complex128, types.complex128))
def f_Hopf(Z, a):
    """The local dynamics function f(z_i) for complex z."""
    Z2 = np.abs(Z)**2
    Z4 = Z2**2
    return a * Z + 2.0 * Z * Z2 - Z * Z4

@njit(types.complex128())
def random_normal_complex():
    return np.random.normal(0.0,1.0) + 1j*np.random.normal(0.0,1.0)

@njit(types.complex128(types.int64,types.complex128[:,:],_type_neighbor_index_list,_type_coupling_strength_list,_type_delay_list))
def sum_oscillator_delayed_input(t,Z,neigh_lst,coupl_lst,delay_lst):
    S = 0.0 + 0.0j
    for j,K,T in zip(neigh_lst,coupl_lst,delay_lst):
        S += K*Z[t-T,j]
    return S


@njit(types.complex128(types.complex128,types.complex128,types.complex128,types.complex128))
def oscillator_step(Z,sum_j_Kij_Zj_delay,sum_j_Kij,a):
    return f_Hopf(Z,a) + sum_j_Kij_Zj_delay - Z*sum_j_Kij

@njit(
    types.complex128[:,:](
        types.float64, types.float64, types.float64, types.int64,
        types.complex128[:], types.complex128[:],
        types.complex128[:,:],     # K_ij
        types.int64[:,:]           # T_ij in *time steps*
    )
)
def integrate_Hopf_oscillators(tTotal, dt, alpha, N, a, Z_initial, J_R_C_matrix, T_delay_matrix):
    # noise dt
    alpha_sqrt_dt = alpha * np.sqrt(dt / 2.0)

    # delay buffer size
    T_max = np.max(T_delay_matrix) + 1

    # creating oscillators and setting IC
    Nt = T_max + types.int64(tTotal / dt) # total number of time steps
    t0 = T_max

    # setting up the previous states all equal to the IC
    Z  = np.zeros((Nt, N), dtype=np.complex128)
    for t in range(T_max):
        Z[t, :] = Z_initial
    
    # sum of coupling
    # Coupling(t) = J * (    sum_j[ Kij * Zj(t-tij)   -     Kij * Zi(t)     ]    )
    #             = J * (    sum_j[ Kij * Zj(t-tij) ] - Zi(t) * sum_j [ Kij ]     )
    # so we make sum_j [ Kij ] in advance
    sum_j_Kij                     = np.sum(J_R_C_matrix, axis=1)
    neigh_lst,coupl_lst,delay_lst = get_coupling_lists(J_R_C_matrix,T_delay_matrix)

    # --- time loop ---
    for t in range(t0,Nt):
        for i in range(N):
            # Euler–Maruyama
            sum_j_Kij_Zj_delay = sum_oscillator_delayed_input(t-1,Z,neigh_lst[i],coupl_lst[i],delay_lst[i])
            Z[t,i]             = Z[t-1,i] + dt * oscillator_step(Z[t-1,i],sum_j_Kij_Zj_delay,sum_j_Kij[i],a[i]) + alpha_sqrt_dt * random_normal_complex()

    return Z[t0:,:]
