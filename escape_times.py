from numba.core.errors import NumbaPerformanceWarning
import warnings

### ignores
### NumbaPerformanceWarning: '@' is faster on contiguous arrays, called on (Array(complex128, 2, 'A', False, aligned=True), Array(complex128
### , 1, 'C', False, aligned=True))
###   delayed_input = K_matrix_ij @ Z_delayed
warnings.filterwarnings(
    "ignore",
    category=NumbaPerformanceWarning
)



import os
import sys
import shlex
import argparse
import scipy.io
import numpy as np
import modules.io as io
import modules.lib_hopf_osc_sim as sim
import modules.lib_escape_times as esc

from joblib import Parallel, delayed

import time
import datetime

def RunSim(cmd_line = ''):

    # for debug
    #sys.argv = 'escape_times.py -ntrials 4 -tTotal 500 -dt 0.1 -alpha 0.1 -beta 0.1 -omega 5 -omega_range 0.01 -lmbda 0.6 -lmbda_range 0.1 -Z0 0 -Z0_std 0 -Z_amp_escape 1 -input_var_FL M_FL -input_var_FN M_FN -input_var_FMRI M_FMRI -input_dir_mat D:/Dropbox/p/pesquisa/epilepsy_criticality/tle_matrix/test -input_type mat -savealltau'.split(' ')


    # if cmd_line is empty,
    # we use sys.argv as input    
    cmd_line,is_argv_modified,sys_argv_temp = _check_cmd_line_sysargv(cmd_line)

    parser   = argparse.ArgumentParser(description="""
Simulate stochastic Hopf oscillators coupled through an empirical connectome,
with heterogeneous local dynamics, time delays derived from tract lengths, and additive noise.

The simulation supports multiple trials, configurable dynamical and noise parameters,
and input connectomes provided either as text files or MATLAB (.mat) files.

Escape times are measured based on a user-defined amplitude threshold of the complex oscillator state.

::: INPUT DATA FORMAT :::
                                       
Input data must be a MAT-file containing the connectivity matrix for each subject,
and node volume if one chooses to normalize parameter lmbda.

Each input data file must have a subject code in its name.
The code for controls is `ddd_d`, and the code for patients is `0ddd_d` (d = digit 0 to 9).

E.g., `mats_301_1.mat` can be the file containing all matrices for control subject `301_1`.
Each matrix must be identified by their corresponding parameters:
`input_var_FL`, `input_var_FN`, `input_var_FMRI`.
I.e., the matrix for fiber length will be read from the variable identified in `input_var_FL`;
the matrix for fiber number will be read from the variable identified in `intput_var_FN`;
the matrix for fMRI will be read from the variable identified in `input_var_FMRI`;
the vector (1 entry per node) for normalizing lmbda will be read from the variable
identified in `input_var_lmbda`.

::: WARNING :::
                                       
The system is solved with a simple Euler-Maruyama method,
so the result is very sensitive to dt.

If the system diverges quickly, try setting a smaller dt,
or a smaller beta.

::: TO-DO LIST :::
                                       
1. add functionality to "record only after all oscillators escaped":
    this can be done with a new integrator function,
    but requires restructuring the delayed variables
    (we can make a list of the delayed variables to save memory, similarly to the neighbor list)
2. measure escape times during integration simulation to save memory.
    This can be integrated with the above functionality

    """, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser            = io.add_simulation_params(parser)
    args              = parser.parse_args()
    simParam          = io.get_simParam_struct_validate_args(args)
    simParam.cmd_line = cmd_line
    
    codes_sorted,fl   = io.get_input_mat_file_list(simParam.input_dir)
    fl_all            = [ f for f in io.select_code_or_die(simParam.input_code,fl) ]
    codes_sorted      = io.get_subject_code(fl_all)
    if simParam.testrun:
        simParam.ntrials = 1
        fl_all           = [fl_all[0]      ] # runs only 1 trial of 1 subject
        codes_sorted     = [codes_sorted[0]] # runs only 1 trial of 1 subject

    # number of subjects
    nsub  = len(fl_all)

    # initializing model parameters
    N                = io.get_input_matrices_mat(input_mat_file=fl_all[0],varname_FL=simParam.input_var_FL,varname_FN=simParam.input_var_FN,varname_FMRI=simParam.input_var_FMRI)[0].shape[0] # number of nodes
    map_lmbda_values = len(simParam.input_var_lmbda) > 0 # if a variable in the input file containing the value of lambda for each subject was supplied, then the lmbda above will be overwritten for each subject

    # initializing output data dictionary
    # output data will be a dict where
    # out['codes']      -> list of subject codes
    # out['simParam']   -> input parameters given to this simulation for reproducibility
    # out['subj_CODE']  -> structtype containing output data for subject identified by code in the list out['codes'] 
    out   = io.structtype(codes=codes_sorted,simParam=simParam,**{ _get_code_output_label(code):io.structtype(tau_mean=None,tau_std=None,tau_samp=None,Pk=None) for code in codes_sorted })

    # simulation running time clocks
    sim_t0 = time.perf_counter()
    sim_ts = sim_t0

    # creating random number generators
    rngseed     = simParam.rngseed if simParam.rngseed > 0 else None
    rng_streams = [
        np.random.default_rng(seed)
        for seed in np.random.SeedSequence(rngseed).spawn(simParam.ntrials)
    ]

    # for each subject
    for l,fname in enumerate(fl_all):
        code             = io.get_subject_code(io.get_filename_no_ext(fname))
        out_label        = _get_code_output_label(code)

        subj_per         = (l+1)/nsub
        print(f' *** simulating subject {l+1}/{nsub} ({100*subj_per:.1f}%) code = ',code)

        M_FL,M_FN,M_FMRI,node_feature_Zscore = io.get_input_matrices_mat(input_mat_file=fname,varname_FL=simParam.input_var_FL,varname_FN=simParam.input_var_FN,varname_FMRI=simParam.input_var_FMRI,varname_lmbda_feature=simParam.input_var_lmbda)
        K                                    = sim.get_coupling_matrix(simParam.beta,M_FN,M_FMRI,normalize_weight=simParam.normalizecoupling)
        T                                    = sim.get_delay_matrix(K,M_FL,simParam.v_tract,simParam.dt) #coupling_delay   = 2.0 / simParam.dt # this is just temporary for testing
        if map_lmbda_values:
            lmbda                            = sim.ZScore_to_lambda(node_feature_Zscore, simParam.lmbda, simParam.lmbda_range, simParam.lmbda_sigmoid_map_k)
        else:
            lmbda                            = sim.get_param_sample(simParam.lmbda, simParam.lmbda_range, N) # this will change for each trial of each subject
        
        # simulating ntrials trials for each subject
        results = Parallel(
            n_jobs=simParam.n_parallel_threads,
            prefer='threads',
            return_as='generator_unordered'
        )(
            delayed(run_trial)(simParam, N, K, T, lmbda, map_lmbda_values, rng)
            for rng in rng_streams
        )

        # initializing output data variables
        n_l               = np.zeros(N,dtype=int)  # number of times that node k is first for subject l
        tau_sum           = np.zeros(N,dtype=float)
        tau2_sum          = np.zeros(N,dtype=float)
        tau_samp          = np.zeros((simParam.ntrials,N),dtype=float) if simParam.savealltau else np.zeros((0,),dtype=float)
        trial_end_counter = 0
        for n, (Z, tau, k) in enumerate(results):
            trial_end_counter += 1
            trial_per          = trial_end_counter/simParam.ntrials
            print(f'     -> trial {trial_end_counter} / {simParam.ntrials} ({100*trial_per:.1f}%) -- completed total: {100*subj_per*trial_per:.1f}%')
            tau_sum  += tau
            tau2_sum += tau**2
            n_l[k]   += 1
            if simParam.savealltau:
                tau_samp[n,:] = tau

        # calculating means over trials for the simulated subject
        out[out_label].tau_mean = tau_sum / simParam.ntrials
        out[out_label].tau_std  = np.sqrt((tau2_sum / simParam.ntrials) - (tau_sum / simParam.ntrials)**2)
        out[out_label].tau_samp = tau_samp.copy()
        out[out_label].Pk       = n_l.astype(float) / simParam.ntrials

        #if simParam.testrun:
        #    break
        
        print(' ------- subject time:', datetime.timedelta(seconds=time.perf_counter()-sim_ts))
        sim_ts = time.perf_counter()

    
    out_fname = io.get_new_file_name(io.get_output_filename(simParam,N=N,suffix='_test' if simParam.testrun else ''))
    if simParam.testrun:
        out[out_label].Z = Z
    print(' *** saving ... ',out_fname)
    #scipy.io.savemat(out_fname,io.convert_items_structtype_to_scipy_struct_array(out),long_field_names=True)
    scipy.io.savemat(out_fname,out.to_dict(),long_field_names=True)

    sim_t1 = time.perf_counter()
    print(' *** ')
    print(' *** simulation time:', datetime.timedelta(seconds=sim_t1-sim_t0))

    # if a cmd_line was provided
    # we have to restore the old sys.argv
    if is_argv_modified:
        sys.argv = sys_argv_temp
    
    return

def _get_code_output_label(code):
    return 'subj_' + code.replace('-','_')

def run_trial(simParam,N,K,T,lmbda,map_lmbda_values,rng):
    omega     = sim.get_param_sample(simParam.omega, simParam.omega_range, N, rng)
    if not map_lmbda_values:
        lmbda = sim.get_param_sample(simParam.lmbda, simParam.lmbda_range, N, rng)
    a         = sim.get_const_param(lmbda,omega)  # oscillator parameter: a = lmbda - 1 + i*omega
    Z0        = sim.get_IC(simParam.Z0,simParam.Z0_std,N, rng)
    Z         = sim.integrate_Hopf_oscillator_network(simParam.tTotal, simParam.dt, simParam.alpha, N,
                                                Z0              = Z0                   , a               = a,
                                                coupling_matrix = K                    , T_delay_matrix  = T,
                                                stop_at_escape  = simParam.stopAtEscape, Z_amp_threshold = simParam.Z_amp_escape,
                                                integrator_type = simParam.integrator)
    # calculating quantities of interest
    tau = esc.calc_escape_time(Z,Z_amp_threshold=simParam.Z_amp_escape,axis=1,dt=simParam.dt)
    k   = np.nanargmin(tau) # finding node that first escaped
    return Z,tau,k

def _check_cmd_line_sysargv(cmd_line):
    is_argv_modified = False
    sys_argv_temp    = []

    if (_var_exists(cmd_line)) and (len(cmd_line) > 0):
        #print('*** using cmdline')
        cmd_line         = _remove_scriptname_from_line(cmd_line)
        sys_argv_temp    = sys.argv
        is_argv_modified = True
        sys.argv         = [_get_scriptname()] + _split_cmd_line(cmd_line)
    else:
        #print('*** using argv')
        if any('ipykernel_launcher' in a for a in sys.argv):
            sys_argv_temp    = sys.argv
            is_argv_modified = True
            sys.argv         = [_get_scriptname()]
        cmd_line         = _remove_scriptname_from_line(' '.join(sys.argv))
    return cmd_line,is_argv_modified,sys_argv_temp

def _get_scriptname():
    return os.path.basename(__file__) #'sorw_simulation.py'

def _remove_scriptname_from_line(cmd_line):
    return cmd_line.replace(_get_scriptname(),'',1).strip() if _var_exists(cmd_line) else ''

def _var_exists(v):
     return not (type(v) is type(None))

def _split_cmd_line(cmd_line):
    # Split using shlex to handle quoted strings and paths with spaces
    tokens = shlex.split(cmd_line)
    
    result = []
    i = 0
    while i < len(tokens):
        if tokens[i].startswith("-"):
            result.append(tokens[i])  # Append argument
            if i + 1 < len(tokens) and not tokens[i + 1].startswith("-"):
                result.append(tokens[i + 1])  # Append value
                i += 1  # Skip the next token (it's already added as value)
        i += 1
    return result

if __name__ == '__main__':
    RunSim()
