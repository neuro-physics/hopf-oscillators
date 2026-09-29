import numpy as np
from modules.io import IntegratorType
from numba import types, njit
from numba.typed import List

import warnings
from numba.core.errors import NumbaExperimentalFeatureWarning

# Filter out the specific experimental feature warning
warnings.filterwarnings("ignore", category=NumbaExperimentalFeatureWarning)


def get_IC(Z0, Z0_std, N, rng=None):
    if type(rng) is type(None):
        return (np.random.normal(Z0.real, Z0_std, N) + 1j * np.random.normal(Z0.imag, Z0_std, N))
    return (rng.normal(Z0.real, Z0_std, N) + 1j * rng.normal(Z0.imag, Z0_std, N))

def get_param_sample(p_mean, p_range, N, rng=None):
    if type(rng) is type(None):
        return p_mean + np.abs(p_range) * np.random.uniform(-1, 1, N)
    return p_mean + np.abs(p_range) * rng.uniform(-1, 1, N)

def get_const_param(lambda_i,omega_i):
    return (lambda_i - 1.0) + 1j * omega_i

def ZScore_to_lambda(z, lambda_mean, lambda_range, k):
    """
    Map z-scores to lambda using a decreasing logistic function.
    Z = 0 (controls) -> lambda_mean
    Z < 0 -> goes to lambda_max = lambda_mean + abs(lambda_range)
    where ideally lambda_mean + abs(lambda_range) ~ 1
    Z > 0 -> goes to lambda_min = lambda_mean - abs(lambda_range)

    Parameters
    ----------
    z : array_like
        Node volume z-scores.
    lambda_mean : float
        mean value of lambda (corresponding to z=0)
    lambda_range : float
        half range within which lambda is mapped: lambda in [lambda_mean - lambda_range; lambda_mean + lambda_range]
    k : float
        Steepness of the sigmoid (> 0).

    Returns
    -------
    lambda_ : ndarray
        Lambda values corresponding to z.
    """
    k          = np.abs(k)
    z          = np.asarray(z)
    lambda_min = lambda_mean - np.abs(lambda_range)
    lambda_max = lambda_mean + np.abs(lambda_range)
    z_c        = np.log((lambda_mean - lambda_min) / (lambda_max - lambda_mean)) / k
    return lambda_min + (lambda_max - lambda_min) / (1.0 + np.exp(k * (z - z_c)))


def get_delay_matrix(K, M_FL, v_tract, dt):
    T    = np.abs(np.round((M_FL / v_tract) / dt).astype(np.int64))
    mask = (K != 0) & (T < 1)
    if np.any(mask):
        print('*** found zero delays, making them equal to 1')
        T[mask] = 1
    return T

def get_coupling_matrix(J,R,C,normalize_weight=False):
    """
    J                -> coupling strength
    R                -> structural coupling matrix (number of fibers)
    C                -> functional coupling matrix (e.g., rs-fMRI correlation matrix)
    normalize_weight -> if set, then beta -> beta(i)=beta/n(i), where n(i) is the number of inputs of node i
    """
    K = R * C # element-wise R * C
    if normalize_weight:
        for i in range(K.shape[0]):
            K[i,:] = J * K[i,:] / K[i,:].sum()
    return np.ascontiguousarray(K.astype(np.complex128))

_type_neighbor_index_list        = types.int64[:]
_type_delay_list                 = types.int64[:]
_type_coupling_strength_list     = types.complex128[:]
_type_all_neighbors_index_list   = types.ListType(_type_neighbor_index_list)
_type_all_delays_list            = types.ListType(_type_delay_list)
_type_all_coupling_strength_list = types.ListType(_type_coupling_strength_list)
_type_get_coupling_lists_output  = types.Tuple((_type_all_neighbors_index_list,_type_all_coupling_strength_list,_type_all_delays_list))
@njit(_type_get_coupling_lists_output(types.complex128[:,:],types.int64[:,:]))
def get_coupling_lists(K,T):
    """
    Construct neighbor, coupling strength, and delay lists from 
    adjacency-like matrices.

    Parameters
    ----------
    K : ndarray (complex128[:, :])
        Coupling strength matrix where nonzero entries indicate 
        coupling strengths between nodes.
    T : ndarray (int64[:, :])
        Delay matrix between connections.

    Returns
    -------
    tuple
        (neighbors_list, coupling_strengths_list, delays_list), where:
        - neighbors_list: list of int64 arrays with neighbor indices
        - coupling_strengths_list: list of complex128 arrays with coupling strength values
        - delays_list: list of int64 arrays with delay values
    """
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

#@njit(types.complex128(types.float64,types.complex128,types.complex128,types.complex128,types.complex128,types.float64))
#def integrate_Euler(dt,Z,sum_j_Kij_Zj_delay,sum_j_Kij,a,alpha_sqrt_dt):
#    return Z + dt * oscillator_step(Z,sum_j_Kij_Zj_delay,sum_j_Kij,a) + alpha_sqrt_dt * random_normal_complex()

@njit(types.complex128(types.complex128, types.complex128))
def f_Hopf(Z, a):
    """The local dynamics function f(z_i) for complex z."""
    Z2 = np.abs(Z)**2
    Z4 = Z2**2
    return a * Z + 2.0 * Z * Z2 - Z * Z4

@njit(types.complex128(types.complex128,types.complex128,types.complex128,types.complex128))
def oscillator_step(Z,sum_j_Kij_Zj_delay,sum_j_Kij,a):
    """
    Compute one step of the Hopf oscillator update.

    Parameters
    ----------
    Z : complex128
        Current oscillator state.
    sum_j_Kij_Zj_delay : complex128
        Delayed coupling total input from neighbors.
    sum_j_Kij : complex128
        Sum of coupling strengths for this oscillator.
    a : complex128
        Oscillator parameter.

    Returns
    -------
    complex128
        Updated oscillator state increment.
    """
    return f_Hopf(Z,a) + sum_j_Kij_Zj_delay - Z*sum_j_Kij

@njit(types.complex128(types.int64,types.complex128[:,:],_type_neighbor_index_list,_type_coupling_strength_list,_type_delay_list))
def sum_oscillator_delayed_input(t,Z,neigh_lst,coupl_lst,delay_lst):
    S = 0.0 + 0.0j
    for j,K,T in zip(neigh_lst,coupl_lst,delay_lst):
        S += K*Z[t-T,j]
    return S

@njit(types.complex128())
def random_normal_complex():
    return np.random.normal(0.0,1.0) + 1j*np.random.normal(0.0,1.0)

@njit(types.complex128(
    types.int64,              # t
    types.float64,            # dt
    types.complex128[:,:],    # Z
    _type_neighbor_index_list,
    _type_coupling_strength_list,
    _type_delay_list,
    types.complex128,         # Z_i
    types.complex128,         # sum_j_Kij
    types.complex128,         # a
    types.float64             # alpha_sqrt_dt
))
def integrate_Euler(
    t, dt, Z,
    neigh_lst, coupl_lst, delay_lst,
    Z_i, sum_j_Kij, a, alpha_sqrt_dt
):
    sum_j_Kij_Zj_delay = sum_oscillator_delayed_input(t, Z, neigh_lst, coupl_lst, delay_lst)
    return Z_i + dt * oscillator_step(Z_i, sum_j_Kij_Zj_delay, sum_j_Kij, a)  + alpha_sqrt_dt * random_normal_complex()

@njit(types.complex128(
    types.int64,              # t
    types.float64,            # dt
    types.complex128[:,:],    # Z
    _type_neighbor_index_list,
    _type_coupling_strength_list,
    _type_delay_list,
    types.complex128,         # Z_i
    types.complex128,         # sum_j_Kij
    types.complex128,         # a
    types.float64              # alpha_sqrt_dt
))
def integrate_RK2(
    t, dt, Z,
    neigh_lst, coupl_lst, delay_lst,
    Z_i, sum_j_Kij, a, alpha_sqrt_dt
):
    # First RK2 evaluation: at t
    sum_j_Kij_Zj_delay_1 = sum_oscillator_delayed_input(t, Z, neigh_lst, coupl_lst, delay_lst)
    sum_j_Kij_Zj_delay_2 = sum_oscillator_delayed_input(t + 1, Z, neigh_lst, coupl_lst, delay_lst)
    F1 = oscillator_step(Z_i,sum_j_Kij_Zj_delay_1,sum_j_Kij,a)
    F2 = oscillator_step(Z_i + dt * F1 + alpha_sqrt_dt * random_normal_complex(),sum_j_Kij_Zj_delay_2,sum_j_Kij,a)
    return Z_i + 0.5 * dt * (F1 + F2) + alpha_sqrt_dt * random_normal_complex()


@njit(
    types.complex128[:,:](
        types.float64, types.float64, types.float64, types.int64,
        types.complex128[:], types.complex128[:],
        types.complex128[:,:],     # K_ij
        types.int64[:,:],          # T_ij in *time steps*
        types.bool_,
        types.float64,
        types.int64 # integrator type
    )
)
def integrate_Hopf_oscillator_network(tTotal, dt, alpha, N, a, Z0, coupling_matrix, T_delay_matrix, stop_at_escape, Z_amp_threshold, integrator_type):
    """
    Integrate a network of Hopf oscillators with delays and noise.

    Parameters
    ----------
    tTotal : float64
        Total simulation time.
    dt : float64
        Time step size.
    alpha : float64
        Noise strength parameter.
    N : int64
        Number of oscillators.
    a : complex128[:]
        Oscillator parameters.
    Z0 : complex128[:]
        Initial states of oscillators.
    coupling_matrix : complex128[:,:]
        Coupling strength matrix.
    T_delay_matrix : int64[:,:]
        Delay matrix in time steps.
    stop_at_escape : boolean
        if True, stops simulation when all oscillators escaped the resting state
    Z_amp_threshold : float64
        amplitude of oscillations used to determine an escape from resting state

    Returns
    -------
    ndarray (complex128[:, :])
        Time evolution of oscillator states.
    """
    if integrator_type == IntegratorType.Euler:
        integration_step = integrate_Euler
    else:
        integration_step = integrate_RK2

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
        Z[t, :] = Z0
    
    # sum of coupling
    # Coupling(t) = J * (    sum_j[ Kij * Zj(t-tij)   -     Kij * Zi(t)     ]    )
    #             = J * (    sum_j[ Kij * Zj(t-tij) ] - Zi(t) * sum_j [ Kij ]     )
    # so we make sum_j [ Kij ] in advance
    sum_j_Kij                     = np.sum(coupling_matrix, axis=1)
    neigh_lst,coupl_lst,delay_lst = get_coupling_lists(coupling_matrix,T_delay_matrix)

    # --- time loop ---
    for t in range(t0,Nt):
        stop_sim = True
        for i in range(N):
            # Euler–Maruyama
            #sum_j_Kij_Zj_delay = sum_oscillator_delayed_input(t-1,Z,neigh_lst[i],coupl_lst[i],delay_lst[i])
            #Z[t,i]             = Z[t-1,i] + dt * oscillator_step(Z[t-1,i],sum_j_Kij_Zj_delay,sum_j_Kij[i],a[i]) + alpha_sqrt_dt * random_normal_complex()
            #Z[t,i]   = integrate_Euler(t-1, dt, Z,neigh_lst[i], coupl_lst[i], delay_lst[i],Z[t-1,i], sum_j_Kij[i], a[i],alpha_sqrt_dt)
            Z[t,i]   = integration_step(t-1, dt, Z,neigh_lst[i], coupl_lst[i], delay_lst[i],Z[t-1,i], sum_j_Kij[i], a[i],alpha_sqrt_dt)
            stop_sim = stop_sim and (np.abs(Z[t,i])>Z_amp_threshold) # stop_sim == True when all oscillators escaped
        
        if stop_sim and stop_at_escape:
            break

    return Z[t0:,:]

def integrate_Hopf_oscillators_const_delay(tTotal, dt, alpha, N, T, a, Z0, coupling_matrix, stop_at_escape, Z_amp_threshold):
    """
    Integrate Hopf oscillators with a constant delay applied to all couplings.

    Parameters
    ----------
    tTotal : float
        Total simulation time.
    dt : float
        Time step size.
    alpha : float
        Noise strength parameter.
    N : int
        Number of oscillators.
    T : int
        Constant delay (in time steps).
    a : complex128[:]
        Oscillator parameters.
    Z0 : complex128[:]
        Initial states of oscillators.
    coupling_matrix : complex128[:,:]
        Coupling strength matrix.

    Returns
    -------
    ndarray (complex128[:, :])
        Time evolution of oscillator states.
    """
    return integrate_Hopf_oscillator_network(tTotal, dt, alpha, N, a, Z0, coupling_matrix, (coupling_matrix>0.0).astype(np.int64)*np.full(coupling_matrix.shape,T,dtype=np.int64), stop_at_escape, Z_amp_threshold)