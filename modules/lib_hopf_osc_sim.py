import numpy as np
from numba import types, njit

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
def f_numba(z, Complex_Param_i):
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
        
        drift_term = f_numba(Z_current, Complex_Param_i) + coupling_term

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


@njit(
    types.complex128[:,:](
        types.float64, types.float64, types.float64, types.int64,
        types.complex128[:], types.complex128[:],
        types.complex128[:,:],     # K_ij
        types.int64[:,:]           # T_ij in *time steps*
    )
)
def integrate_Hopf_oscillators(
    T, dt, alpha, N,
    Z_initial, Complex_Param_i,
    K_matrix_ij, T_steps_ij
):

    Nt = types.int64(T / dt)

    # --- delay buffer size ---
    max_delay = np.max(T_steps_ij)
    buffer_size = max_delay + 1

    sqrt_dt_complex_noise = np.sqrt(dt / 2.0)

    # --- storage ---
    Z = np.zeros((Nt, N), dtype=np.complex128)
    Z[0, :] = Z_initial

    history_buffer = np.zeros((buffer_size, N), dtype=np.complex128)
    for k in range(buffer_size):
        history_buffer[k, :] = Z_initial

    history_index = 0

    # --- row sums of K ---
    row_sums_K = np.sum(K_matrix_ij, axis=1)

    # --- time loop ---
    for t in range(Nt - 1):
        Z_current = Z[t, :]

        # ----- delayed coupling term -----
        delayed_input = np.zeros(N, dtype=np.complex128)

        for i in range(N):
            acc = 0.0 + 0.0j
            for j in range(N):
                delay_ij = T_steps_ij[i, j]
                buf_idx = (history_index - delay_ij) % buffer_size
                acc += K_matrix_ij[i, j] * history_buffer[buf_idx, j]
            delayed_input[i] = acc

        current_dependent_term = Z_current * row_sums_K
        coupling_term = delayed_input - current_dependent_term
        # ----------------------------------

        drift_term = f_numba(Z_current, Complex_Param_i) + coupling_term

        # noise
        dW_R = np.random.normal(0.0, 1.0, N) * sqrt_dt_complex_noise
        dW_I = np.random.normal(0.0, 1.0, N) * sqrt_dt_complex_noise
        noise_term = alpha * (dW_R + 1j * dW_I)

        # Euler–Maruyama
        Z[t + 1, :] = Z_current + drift_term * dt + noise_term

        # update history
        history_index = (history_index + 1) % buffer_size
        history_buffer[history_index, :] = Z[t + 1, :]

    return Z
