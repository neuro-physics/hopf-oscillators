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
                                       
Each input file (`txt` or `mat`) is supposed to have a subject code in its name.
The code for controls is `ddd_d`, and the code for patients is `0ddd_d` (d = digit).
The same code must be on the corresponding files for each matrix.

For example, for `txt` input, we can have `FL_301_1.txt`,
`FN_301_1.txt`, `FMRI_301_1.txt` as the names of the input files containing the fiber length (FL),
fiber number (FN) and rs-fMRI (FMRI) matrices for the control subject identified by code `301_1`.

Similarly for MAT-file inputs, but then only one file with all matrices inside is needed.
E.g., `mats_301_1.mat` can be the file containing all matrices control subject `301_1`.
Each matrix must be identified by their corresponding
parameters: `input_var_FL`, `input_var_FN`, `input_var_FMRI`.

::: WARNING :::
The system is solved with a simple Euler-Maruyama method,
so the result is very sensitive to dt.

If the system diverges quickly, try setting a smaller dt,
or a smaller beta.
    """, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser            = io.add_simulation_params(parser)
    args              = parser.parse_args()
    simParam          = io.get_simParam_struct_validate_args(args)
    simParam.cmd_line = cmd_line

    is_mat_input      = simParam.input_type == 'mat'

    if is_mat_input:
        codes_sorted,fl             = io.get_input_mat_file_list(simParam.input_dir_mat)
        fl_all                      = [ (f,) for f in io.select_code_or_die(simParam.input_code,fl) ]
        get_input_matrices          = io.get_input_matrices_mat
        get_args_get_input_matrices = lambda *fnames: dict(input_mat_file=fnames[0],varname_FL=simParam.input_var_FL,varname_FN=simParam.input_var_FN,varname_FMRI=simParam.input_var_FMRI)
    else:
        codes_sorted,fl_FL, fl_FN, fl_FMRI  = io.get_input_txt_file_lists(simParam.input_dir_FL,simParam.input_dir_FN,simParam.input_dir_FMRI)
        fl_all                              = [ (f1,f2,f3) for f1,f2,f3 in zip(io.select_code_or_die(simParam.input_code,fl_FL), io.select_code_or_die(simParam.input_code,fl_FN), io.select_code_or_die(simParam.input_code,fl_FMRI))  ]
        get_input_matrices                  = io.get_input_matrices_txt
        get_args_get_input_matrices         = lambda *fnames: dict(input_txt_file_FL=fnames[0],input_txt_file_FN=fnames[1],input_txt_file_FMRI=fnames[2])


    if simParam.testrun:
        simParam.ntrials = 1
        fl_all           = [fl_all[0]]
        codes_sorted     = [codes_sorted[0]]

    nsub  = len(fl_all)
    N     = get_input_matrices(**get_args_get_input_matrices(*fl_all[0]))[0].shape[0] # number of nodes
    omega = sim.get_param_sample(simParam.omega, simParam.omega_range,N)
    lmbda = sim.get_param_sample(simParam.lmbda, simParam.lmbda_range,N)
    a     = sim.get_const_param(lmbda,omega) # a = lmbda - 1 + i*omega
    out   = dict(codes=codes_sorted,simParam=simParam,**{ code:io.structtype(tau_mean=None,tau_std=None,tau_samp=None,Pk=None) for code in codes_sorted })

    sim_t0 = time.perf_counter()
    sim_ts = sim_t0
    

    # for each subject
    for l,fnames in enumerate(fl_all):
        code             = io.get_subject_code(io.get_filename_no_ext(fnames[0]))
        subj_per         = (l+1)/nsub
        print(f' *** simulating subject {l+1}/{nsub} ({100*subj_per:.1f}%) code = ',code)
        M_FL,M_FN,M_FMRI = get_input_matrices(**get_args_get_input_matrices(*fnames))
        K                = sim.get_coupling_matrix(simParam.beta,M_FN,M_FMRI,normalize_weight=simParam.normalizecoupling)
        T                = np.round((M_FL / simParam.v_tract) / simParam.dt).astype(np.int64) #coupling_delay   = 2.0 / simParam.dt # this is just temporary for testing
        n_l              = np.zeros(N,dtype=int)  # number of times that node k is first for subject l
        tau_sum          = np.zeros(N,dtype=float)
        tau2_sum         = np.zeros(N,dtype=float)
        tau_samp         = np.zeros((0,),dtype=float)
        if simParam.savealltau:
            tau_samp     = np.zeros((simParam.ntrials,N),dtype=float)
        for n in range(simParam.ntrials):
            trial_per = (n+1)/simParam.ntrials
            print(f'     -> trial {n+1} / {simParam.ntrials} ({100*trial_per:.1f}%) -- completed total: {100*subj_per*trial_per:.1f}%')
            Z_initial = sim.get_IC(simParam.Z0,simParam.Z0_std,N)
            Z         = sim.integrate_Hopf_oscillators(simParam.tTotal, simParam.dt, simParam.alpha, N, Z_initial=Z_initial, a=a, J_R_C_matrix=K, T_delay_matrix=T)
            tau       = esc.calc_escape_time(Z,Z_amp_threshold=simParam.Z_amp_escape,axis=1,dt=simParam.dt)
            k         = np.nanargmin(tau)
            tau_sum  += tau
            tau2_sum += tau**2
            n_l[k]   += 1
            if simParam.savealltau:
                tau_samp[n,:] = tau.copy()
            if simParam.testrun:
                break
        out[code].tau_mean = tau_sum / simParam.ntrials
        out[code].tau_std  = np.sqrt((tau2_sum / simParam.ntrials) - (tau_sum / simParam.ntrials)**2)
        out[code].tau_samp = tau_samp.copy()
        out[code].Pk       = n_l.astype(float) / simParam.ntrials
        if simParam.testrun:
            break
        
        print(' ------- subject time:', datetime.timedelta(seconds=time.perf_counter()-sim_ts))
        sim_ts = time.perf_counter()

    
    out_fname = io.get_new_file_name(io.get_output_filename(simParam,N=N,suffix='_test' if simParam.testrun else ''))
    if simParam.testrun:
        out[code].Z = Z
    print(' *** saving ... ',out_fname)
    scipy.io.savemat(out_fname,out)

    sim_t1 = time.perf_counter()
    print(' *** ')
    print(' *** simulation time:', datetime.timedelta(seconds=sim_t1-sim_t0))

    # if a cmd_line was provided
    # we have to restore the old sys.argv
    if is_argv_modified:
        sys.argv = sys_argv_temp
    
    return

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
